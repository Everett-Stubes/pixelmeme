Baked idle clips go here as `honnold.pxv.npz`.

Nothing is shipped in this directory: supply your own footage and bake it with

    python tools/prepare_video.py your-clip.mp4 --seconds 45

If this directory stays empty the app generates an endless pixel-art climb
instead, so it still has something to show.
