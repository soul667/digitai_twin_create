```shell
curl -fsSL https://bun.com/install | bash
```

`gs_vis/visionary/` is reserved for the Visionary viewer submodule.
The wrapper scripts in `gs_vis/` do not write extra files into the submodule.

## Local 3DGS viewer

Use the helper script below to visualize the current exported Gaussian PLY:

```shell
./gs_vis/run_visionary.sh
```

By default it loads:

```shell
./da3/output_gs/gs_ply/0000.ply
```

Then open:

```shell
http://localhost:3000/demo/simple/index.html
```

To visualize another PLY:

```shell
MODEL_SRC=/abs/path/to/your_model.ply ./gs_vis/run_visionary.sh
```

Then click `选择文件` or drag the `.ply` into the official Visionary page.

## Docker Desktop Mode

To run Visionary inside a Selkies EGL remote desktop with one GPU mounted:

```shell
./gs_vis/run_visionary_vnc.sh
```

This starts:

- Selkies remote desktop on `http://localhost:8080`
- Visionary dev server on `http://localhost:3000`

The container will open Chrome inside the remote desktop and load:

```shell
http://127.0.0.1:3000/demo/simple/index.html
```

To visualize another PLY:

```shell
MODEL_SRC=/abs/path/to/your_model.ply ./gs_vis/run_visionary_vnc.sh
```

Inside the VNC browser, load the model from:

```shell
/workspace/da3/output_gs/gs_ply/0000.ply
```

or the path provided by `MODEL_SRC`.

To choose which single GPU is mounted into the container:

```shell
GPU_DEVICE=1 ./gs_vis/run_visionary_vnc.sh
```

To set the Selkies login password:

```shell
PASSWD=your_password ./gs_vis/run_visionary_vnc.sh
```

If the host NVIDIA driver version differs from the default baked image version, pass it explicitly when building/running:

```shell
NVIDIA_DRIVER_VERSION=575.64.03 ./gs_vis/run_visionary_vnc.sh
```

The current Docker setup uses `ghcr.io/selkies-project/nvidia-egl-desktop:22.04`, which is a better fit when the host already has its own Xorg/desktop session running. It also mounts `/dev/dri` and defaults `VGL_DISPLAY=egl`.

## GitHub image publishing

The workflow [publish-visionary-vnc.yml](/data2/axgu/code/digitai_twin_create/.github/workflows/publish-visionary-vnc.yml) builds the VNC image from [Dockerfile](/data2/axgu/code/digitai_twin_create/gs_vis/docker/Dockerfile) and publishes it to GHCR on GitHub Release publish.

Published image:

```shell
ghcr.io/<owner>/digitai-twin-create-visionary-vnc
```
111
