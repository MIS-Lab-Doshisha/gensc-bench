#!/bin/bash

source mrtrix_env.sh

MAX_JOBS=5
THREADS_PER_JOB=6
STAGGER_SEC=180
export MRTRIX_NTHREADS=$THREADS_PER_JOB

batch_index="${1:-0}"
batch_name=$(printf "batch_%03d" "$batch_index")
batch_dir="hcp/${batch_name}"
atlas="path/to/atlas/AAL116.nii.gz"

target_dir="hcp/${batch_name}/raw"
work_root="hcp/${batch_name}/work"

output_dir="hcp/output"
mkdir -p "$output_dir"

log_dir="${batch_dir}/logs"
mkdir -p "$log_dir"

run_date="$(date +%Y-%m-%d_%H-%M-%S)"
log_file="${log_dir}/pipeline_${run_date}.log"

skip_if_exists() {
    local file_path="$1"
    shift

    if [ ! -s "$file_path" ]; then
        "$@"
    else
        echo "Skipping command as file exists: $file_path"
    fi
}

process_subject() (
    set -e

    raw_dir="$1"
    subject_id="$2"
    work_dir="$3"

    echo "Processing subject: $subject_id"

    skip_if_exists "${work_dir}/dwi.mif" \
    mrtrix mrconvert "${raw_dir}/T1w/Diffusion/data.nii.gz" "${work_dir}/dwi.mif" \
    -fslgrad "${raw_dir}/T1w/Diffusion/bvecs" "${raw_dir}/T1w/Diffusion/bvals" \
    -datatype float32
    skip_if_exists "${work_dir}/dwi_mask.mif" \
    mrtrix mrconvert "${raw_dir}/T1w/Diffusion/nodif_brain_mask.nii.gz" "${work_dir}/dwi_mask.mif" -datatype bit

    skip_if_exists "${work_dir}/wm_response.txt" \
    mrtrix dwi2response dhollander "${work_dir}/dwi.mif" "${work_dir}/wm_response.txt" "${work_dir}/gm_response.txt" "${work_dir}/csf_response.txt" -mask "${work_dir}/dwi_mask.mif"

    skip_if_exists "${work_dir}/wmfod.mif" \
    mrtrix dwi2fod msmt_csd "${work_dir}/dwi.mif" \
    "${work_dir}/wm_response.txt" "${work_dir}/wmfod.mif" \
    "${work_dir}/gm_response.txt" "${work_dir}/gm.mif" \
    "${work_dir}/csf_response.txt" "${work_dir}/csf.mif" \
    -mask "${work_dir}/dwi_mask.mif"

    skip_if_exists "${work_dir}/wmfod_norm.mif" \
    mrtrix mtnormalise \
    "${work_dir}/wmfod.mif" "${work_dir}/wmfod_norm.mif" \
    "${work_dir}/gm.mif" "${work_dir}/gm_norm.mif" \
    "${work_dir}/csf.mif" "${work_dir}/csf_norm.mif" \
    -mask "${work_dir}/dwi_mask.mif"

    skip_if_exists "${work_dir}/5TT.mif" \
    mrtrix 5ttgen fsl "${raw_dir}/T1w/T1w_acpc_dc_restore_brain.nii.gz" "${work_dir}/5TT.mif" -premasked

    skip_if_exists "${work_dir}/tracks_10m.tck" \
    mrtrix tckgen "${work_dir}/wmfod_norm.mif" "${work_dir}/tracks_10m.tck" \
    -algorithm iFOD2 \
    -act "${work_dir}/5TT.mif" \
    -backtrack -crop_at_gmwmi -seed_dynamic "${work_dir}/wmfod_norm.mif" \
    -select 10000000

    skip_if_exists "${work_dir}/sift2_weights_10m.txt" \
    mrtrix tcksift2 "${work_dir}/tracks_10m.tck" "${work_dir}/wmfod_norm.mif" "${work_dir}/sift2_weights_10m.txt" \
    -act "${work_dir}/5TT.mif" \
    -out_mu "${work_dir}/out_mu_10m.txt"

    skip_if_exists "${work_dir}/AAL116_subject.nii.gz" \
    mrtrix applywarp \
    --in=${atlas} \
    --ref=${raw_dir}/T1w/T1w_acpc_dc_restore_brain.nii.gz \
    --warp=${raw_dir}/MNINonLinear/xfms/standard2acpc_dc.nii.gz \
    --out=${work_dir}/AAL116_subject.nii.gz \
    --interp=nn \
    --datatype=int

    skip_if_exists "${work_dir}/sc_aal116.csv" \
    mrtrix tck2connectome ${work_dir}/tracks_10m.tck \
    ${work_dir}/AAL116_subject.nii.gz \
    ${work_dir}/sc_aal116.csv \
    -tck_weights_in ${work_dir}/sift2_weights_10m.txt \
    -scale_invnodevol \
    -symmetric \
    -zero_diagonal \
    -out_assignments ${work_dir}/assignment.txt
)


pipeline_start="$(date +%s)"
echo "============================================================"
echo "Pipeline started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

job_index=0

for raw_dir in $target_dir/*/; do
    [ -d "$raw_dir" ] || continue

    subject_id="$(basename "$raw_dir")"
    work_dir="${work_root}/${subject_id}"
    subject_log="${log_dir}/${subject_id}_${run_date}.log"

    mkdir -p "$work_dir"

    if [ -s "${work_dir}/sc_aal116.csv" ]; then
        echo "Subject ${subject_id} already processed. Skipping."
        continue
    fi

    while [ $(jobs -rp | wc -l) -ge $MAX_JOBS ]; do
        sleep 10
    done

    (
        exec > >(tee -a "$subject_log") 2>&1

        if [ ${job_index} -lt "$MAX_JOBS" ]; then
            echo "SUBJECT: $subject_id, DELAY: $((job_index * STAGGER_SEC)) seconds"

            sleep $((job_index * STAGGER_SEC))
        fi

        subject_starts="$(date +%s)"

        process_subject "$raw_dir" "$subject_id" "$work_dir"
        status=$?

        subject_ends="$(date +%s)"
        subject_elapsed=$((subject_ends - subject_starts))
        subject_elapsed_min=$((subject_elapsed / 60))
        subject_elapsed_sec=$((subject_elapsed % 60))

        if [ $status -eq 0 ]; then
            echo
            echo "[SUCCESS] Subject ${subject_id} processed successfully"
            echo "Elapsed time: ${subject_elapsed_min}m.${subject_elapsed_sec}s"
            echo
        else
            echo
            echo "[ERROR] Subject ${subject_id} failed to process"
            echo "Elapsed time: ${subject_elapsed_min}m.${subject_elapsed_sec}s"
            echo
            continue
        fi

        sc_file="${work_dir}/sc_aal116.csv"

        if [ ! -s "$sc_file" ]; then
            echo "[ERROR] SC file not found for subject ${subject_id}"
            continue
        fi

        if [ ! -s "${output_dir}/${subject_id}_sc_aal116.csv" ]; then
            awk \
            -F',' '
                BEGIN { error=0 }
                {
                    if (NF != 116) {
                        printf "ERROR: row %d has %d columns (expected 116)\n", NR, NF
                        error=1
                    }

                    nonzero_count=0

                    for (i=1; i<=NF; i++) {
                        value = $i

                        lower=tolower(value)
                        if (lower ~ /nan/ || lower ~ /inf/) {
                            printf "ERROR: row %d, column %d has invalid value: %s\n", NR, i, value
                            error=1
                        }

                        if (value != 0) {
                            nonzero_count++
                        }
                    }

                    if (nonzero_count == 0) {
                            printf "ERROR: row %d has all zero values\n", NR
                            error=1
                    }
                }
                END {
                    if (NR != 116) {
                        printf "ERROR: file has %d rows (expected 116)\n", NR
                        error=1
                    }

                    if (error) {
                        exit 1
                    } else {
                        print "SC QC passed"
                    }
                }
            ' "$sc_file"

            qc_status=$?

            if [ $qc_status -ne 0 ]; then
                echo "[ERROR] SC QC failed for subject ${subject_id}"
            else
                echo "[SUCCESS] SC QC passed for subject ${subject_id}"
                cp ${work_dir}/sc_aal116.csv ${output_dir}/${subject_id}_sc_aal116.csv
            fi
        fi
    ) &

    job_index=$((job_index + 1))
done

wait

pipeline_end="$(date +%s)"
pipeline_elapsed=$((pipeline_end - pipeline_start))

elapsed_min=$((pipeline_elapsed / 60))
elapsed_sec=$((pipeline_elapsed % 60))

{
    echo "============================================================"
    echo "Pipeline completed: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Max_jobs: $MAX_JOBS, Threads per job: $THREADS_PER_JOB"
    echo "Total elapsed time: ${elapsed_min}m.${elapsed_sec}s."
    echo "============================================================"
} | tee -a "$log_file"
