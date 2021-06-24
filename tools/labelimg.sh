podman run -it --rm -v $(pwd)/images:/images:z -v $(pwd)/annotations:/annotations:z \
	--network host -e DISPLAY=0:0 ludwigprager/labelimg:1

chown -R $(id -un):$(id -un) $(pwd)/images
chown -R $(id -un):$(id -un) $(pwd)/annotations
