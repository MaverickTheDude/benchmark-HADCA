# exampleCP

simple numerical example with cart + #N pendulums
HDCA + AHDCA test case

Compile with:

meson setup builddir (tylko raz, zeby ustalic folder z rozwiazaniem/binarka)

cd builddir

meson compile


# Develop with Docker

This piece of work was created before diving into *Docker best practices*, excuse me.

## I have **not** Docker in my WSL 2 (Ubuntu specific)

Please follow this [excellent blog post](https://dev.to/bowmanjd/install-docker-on-windows-wsl-without-docker-desktop-34m9) by Jonathan Bowman to fully understand what is going under the hood. I've copied necessary steps at the bottom of the page.

## I have Docker in my WSL 2

1. Install VS Code extension [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
2. Once you've got *exemplecp* project opened (e.g. use `code .` in WSL in *exemplecp* path) in VS Code, click <kbd>F1</kbd> and find option `Remote-Containers: Open Folder in Container ...`.
3. Leave the default folder (should point to *examplecp* folder) and click `OK`.
4. If it is the first time you launch the container, it will take some time building.

### Other tricks

- If you change the Dockerfile or other Docker specific configurations, run `Remote-Containers: Rebuild Container`.
- You can check container logs by `Remote-Containers: Show Container Logs...`.
- The container is configured so that you can debug and use run options from VS Code IDE.


## Stages

### Development stage (*dev*)

This stage is primarly designed to develop in VS Code with [Remote Containers (GitHub)](https://github.com/microsoft/vscode-dev-containers) extension. Instructions attached above.

### Building stage (*build*)

You can build the building container image as follows
```
    docker build --target build -t examplecp/build -f ./.devcontainer/Dockerfile .
```

To run the building container (with binaries already compiled according to meson.build) run the container
```
    docker run --rm -it examplecp/build sh
```

### Release stage (*release*)

```
    docker build --target release -t examplecp/release -f ./.devcontainer/Dockerfile .
```

To run the release container
```
    docker run --rm -it examplecp/release
```

## Useful basic Docker commands
- `docker images -a`
- `docker ps -a`
- `docker rmi $(docker images -a -q)`

## Useful links
- [Visual Studio Code - Containers in WSL](https://code.visualstudio.com/blogs/2020/07/01/containers-wsl)
- [Visual Studio Code - Docs on containers](https://code.visualstudio.com/docs/remote/containers)
- [GitHub repository of C++ template for containers in VS Code](https://github.com/microsoft/vscode-dev-containers/tree/v0.209.6/containers/cpp)
- [Tutorial on installing pure Docker in WSL 2](https://dev.to/bowmanjd/install-docker-on-windows-wsl-without-docker-desktop-34m9)
- [Docker best practices](https://docs.docker.com/develop/dev-best-practices/)
- [Docker basics tutorial](https://docker-curriculum.com/)
- [Docker multistage build](https://docs.docker.com/develop/develop-images/multistage-build/)
- [Remote - Containers project configuration](https://code.visualstudio.com/docs/remote/devcontainerjson-reference)

## Docker short installation guide
1. Install dependencies `sudo apt install --no-install-recommends apt-transport-https ca-certificates curl gnupg2`.
2. Set some OS-specific variables `source /etc/os-release`.
3. Make sure `apt` will trust the repo `curl -fsSL https://download.docker.com/linux/${ID}/gpg | sudo apt-key add -`.
4. Update repo information `echo "deb [arch=amd64] https://download.docker.com/linux/${ID} ${VERSION_CODENAME} stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt update`.
5. Install docker `sudo apt install docker-ce docker-ce-cli containerd.io`.
6. Add your user to `Docker` group `sudo usermod -aG docker $USER`.
7. Prepare directory for the docker
```
    DOCKER_DIR=/mnt/wsl/shared-docker
    mkdir -pm o=,ug=rwx "$DOCKER_DIR"
    chgrp docker "$DOCKER_DIR"
```
8. Create `sudo mkdir /etc/docker/` and place a config file `/etc/docker/daemon.json` with the following content
```
    {
        "hosts": ["unix:///mnt/wsl/shared-docker/docker.sock"]
    }
```
9. You can launch Docker daemon `sudo dockerd` and try 
```
    docker -H unix:///mnt/wsl/shared-docker/docker.sock run --rm hello-world
```
10. Add to your `.bashrc` too autolaunch `dockerd` deamon
```
DOCKER_DISTRO="Ubuntu-20.04"
DOCKER_DIR=/mnt/wsl/shared-docker
DOCKER_SOCK="$DOCKER_DIR/docker.sock"
export DOCKER_HOST="unix://$DOCKER_SOCK"
if [ ! -S "$DOCKER_SOCK" ]; then
    mkdir -pm o=,ug=rwx "$DOCKER_DIR"
    chgrp docker "$DOCKER_DIR"
    /mnt/c/Windows/System32/wsl.exe -d $DOCKER_DISTRO sh -c "nohup sudo -b dockerd < /dev/null > $DOCKER_DIR/dockerd.log 2>&1"
fi
```
11. Add `%docker ALL=(ALL)  NOPASSWD: /usr/bin/dockerd` in `sudo visudo` to omit password checking on `dockerd` launching.