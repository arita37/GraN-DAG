1. Install Debian Dependencies:
```


sudo apt-get update && sudo apt-get install -y \
    build-essential \
    libssl-dev \
    uuid-dev \
    libgpgme11-dev \
    squashfs-tools     




```

2. Install Go
```

Need to remove manually
/usr/local/go/


export VERSION=1.11 OS=linux ARCH=amd64
cd /tmp
wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz
sudo tar -C /usr/local -xzf go$VERSION.$OS-$ARCH.tar.gz
echo 'export GOPATH=${HOME}/go' >> ~/.bashrc
echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> ~/.bashrc
source ~/.bashrc



```

3. Install Singularity
```
mkdir -p $GOPATH/src/github.com/sylabs
cd $GOPATH/src/github.com/sylabs
git clone https://github.com/sylabs/singularity.git
cd singularity
```

4. Install Go dependencies
```
go get -u -v github.com/golang/dep/cmd/dep
```

5. Compile Singular Binary
```
https://github.com/sylabs/singularity/issues/4765

sudo apt-get install uuid-dev
sudo apt-get ssl


cd $GOPATH/src/github.com/sylabs/singularity
./mconfig
make -C builddir
sudo make -C builddir install



```

6. Clone code repository
```
cd $GOPATH/src/github.com


git clone https://github.com/kurowasan/GraN-DAG.git
cd GraN-DAG




```

7. Extract the dataset you want to train on. Update the dataset name in the work.sh if you want to train on different dataset
```
cd data
unzip data_p10_e10_n1000_GP.zip
cd ..
```

8. To run the algorithm, run the start_example.sh with updated paths.
```
gedit start_example.sh
CODE_PATH="~/go/src/github.com/GraN-DAG"
EXP_PATH="~/go/src/github.com/GraN-DAG/experiments"
DATA_PATH="~/go/src/github.com/GraN-DAG/data/data_p10_e10_n1000_GP"
./start_example.sh


```

9. To simulate the baseline results, uncomment the desired algorithm in start_example.sh and rerun 8.





