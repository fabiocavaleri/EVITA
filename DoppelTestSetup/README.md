# Instruction to setup a DoppelTest installation in Docker

Issue: The available code assume that the host (running the python code for generating scenarios) can communicate directly with the Docker instances running the Apollo Planner.

Idea: Run DoppleTest inside a Docker container (using docker in docker). 

Problem: Existing Apollo's script cannot start containers from within a container. So we need to workaround this.

Solution:
    1 - Prestart all the containers using the ""start all containers script". This script starts Apollo's instances and map DreaView network port on the host (starting from '8888' on)
    2 - Disable container startup in the python code
    3 - Run the GA insider the docker container

## Manually start the Apollo in Docker and start DreamView on them (Only once before starting the testing, but after having configured Apollo as per DoppelTest instructions)

Assuming <APOLLO_ROOT> points to the root of `BaiduApollo_DoppelTest` 
Replace the script `<APOLLO_ROOT>/docker/scripts/dev_start.sh` with the one in this folder `dev_start.sh.patched`:

```
cp dev_start.sh.patched <APOLLO_ROOT>/docker/scripts/
cd <APOLLO_ROOT>/docker/scripts/
mv dev_start.sh dev_start.sh.original
mv dev_start.sh.patched dev_start.sh
```

`dev_start.sh.patched` is the same script with an additional parameter (`-p`) to map the docker port of dreamview (8888) on a specified host port.

After you configured apollo according to DoppleTest instruction, start Apollo in Docker as follows:

```
USER="ROUTE_0"; ./dev_start.sh -l -y -p 8888
USER="ROUTE_1"; ./dev_start.sh -l -y -p 8889
USER="ROUTE_2"; ./dev_start.sh -l -y -p 8890
```

This will start three instances each opening a different port.

Manually start DreamViewr on each of those docker containers:

```
docker exec apollo_dev_ROUTE_0 /bin/bash ./scripts/bootstrap.sh start
docker exec apollo_dev_ROUTE_1 /bin/bash ./scripts/bootstrap.sh start
docker exec apollo_dev_ROUTE_2 /bin/bash ./scripts/bootstrap.sh start
```

Check that everything is find by opening the following links into the browser:

[http://localhost:8888/](http://localhost:8888/)
[http://localhost:8889/](http://localhost:8889/)
[http://localhost:8890/](http://localhost:8890/)

## Disable Starting the containers in the python code

The following shows how to patch main_ga.py. Patching test_main.py and the other ga files should be the same, as they all start N containers at the beginning, and then start the test generation.


Open `main_ga.py` and comment out lines 280 - 283:

```
    #for ctn in containers:
    #    ctn.start_instance()
    #    ctn.start_dreamview()
    #    print(f'Dreamview at http://{ctn.ip}:{ctn.port}')
```

Since we started only 3 docker containers, we need to edit `config.py` and set `MAX_ADC_COUNT = 3`
Since we are going to start the python code insider a docker container and <APOLLO_ROOT> will be mapped to `/BaiduApollo_DoppelTest` set `APOLLO_ROOT = "/BaiduApollo_DoppelTest"`

## Run the main_ga.py withing a docker container

Since the python code needs to directly interact with the Apollos running inside docker container, we need to start the python code from within a new containers.
Python 3.8 is required, so we need to select that image. I did not have time to setup a proper DockerFile, so we setup stuff manually

Assuming you have <APOLLO_ROOT> and <DOPPEL_TEST_ROOT> one beside the other, invoke the following command:

```
docker run -it -v /var/run/docker.sock:/var/run/docker.sock -v <DOPPEL_TEST_ROOT>:/DoppelTest -v .<APOLLO_ROOT>:/BaiduApollo_DoppelTest python:3.8 /bin/bash
```

Inside the docker container just started:

Install the docker cli command:
```
apt-get update
apt-get install docker.io -y
```

Setup the python dependencies:

```
cd DoppelTest
pip install --upgrade pip
pip install -r requirements.txt
pip install requests==2.29.0
```

Run the main algorithm:

```
python main_ga.py
```

If you have the browsers open on the DreaView, you should see the vehicles started and the console should show the results. According to DoppelTest configuration, you'll see results and scenarios under the results folder (on your host)
