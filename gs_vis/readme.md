```shell
curl -fsSL https://bun.com/install | bash
```

`gs_vis/visionary/` is reserved for the Visionary viewer submodule.

## Local 3DGS viewer

Use the helper script below to visualize the current exported Gaussian PLY:

```shell
./gs_vis/run_visionary.sh
```

By default it loads:

```shell
./da3/output_gs/gs_ply/0000.ply
```

and copies it to:

```shell
./gs_vis/visionary/models/current_output.ply
```

Then open:

```shell
http://localhost:3000/demo/simple/current_output.html
```

To visualize another PLY:

```shell
MODEL_SRC=/abs/path/to/your_model.ply ./gs_vis/run_visionary.sh
```

## Docker VNC mode

To run Visionary inside a Docker desktop with VNC/noVNC and one GPU mounted:

```shell
./gs_vis/run_visionary_vnc.sh
```

This starts:

- Visionary dev server on `http://localhost:3000`
- VNC on `localhost:5901`
- noVNC on `http://localhost:6080/vnc.html`

The container will open Chromium inside the virtual desktop and load:

```shell
http://127.0.0.1:3000/demo/simple/current_output.html
```

To visualize another PLY:

```shell
MODEL_SRC=/abs/path/to/your_model.ply ./gs_vis/run_visionary_vnc.sh
```

To choose which single GPU is mounted into the container:

```shell
GPU_DEVICE=1 ./gs_vis/run_visionary_vnc.sh
```

## GitHub image publishing

The workflow [publish-visionary-vnc.yml](/data2/axgu/code/digitai_twin_create/.github/workflows/publish-visionary-vnc.yml) builds the VNC image from [Dockerfile](/data2/axgu/code/digitai_twin_create/gs_vis/docker/Dockerfile) and publishes it to GHCR on GitHub Release publish.

Published image:

```shell
ghcr.io/<owner>/digitai-twin-create-visionary-vnc
```
