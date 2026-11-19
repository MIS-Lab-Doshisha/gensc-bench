mrtrix() {
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -e HOME=/tmp \
    -v $HOME/data:/data \
    -e DISPLAY="$DISPLAY" \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -w /data \
    -e FSLDIR=/opt/fsl \
    -e MRTRIX_NTHREADS \
    -e PATH=/opt/mrtrix3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/fsl/share/fsl/bin:/opt/fsl/bin  \
    mrtrix3/mrtrix3:3.0.8 \
    "$@"
}