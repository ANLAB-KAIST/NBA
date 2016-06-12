#! /bin/bash

unset DEBUG
mkdir -p bin-backup

if [ "$1" == "knapp" ]; then
    echo "Compiling for Knapp experiments..."

    export NBA_USE_CUDA=0
    export NBA_USE_KNAPP=1

    # Knapp does not support datablock reusing.

    export NBA_BATCHING_SCHEME=2
    export NBA_BRANCHPRED_SCHEME=0
    export NBA_USE_OPENSSL_EVP=1
    export NBA_REUSE_DATABLOCKS=0
    snakemake clean
    snakemake -j
    cp bin/main bin-backup/main
    cp bin/main bin-backup/main.branchpred.off

    export NBA_BATCHING_SCHEME=2
    export NBA_BRANCHPRED_SCHEME=1
    export NBA_USE_OPENSSL_EVP=1
    export NBA_REUSE_DATABLOCKS=0
    snakemake clean
    snakemake -j
    cp bin/main bin-backup/main.branchpred.on

    export NBA_BATCHING_SCHEME=2
    export NBA_BRANCHPRED_SCHEME=2
    export NBA_USE_OPENSSL_EVP=1
    export NBA_REUSE_DATABLOCKS=0
    snakemake clean
    snakemake -j
    cp bin/main bin-backup/main.branchpred.always

    export NBA_BATCHING_SCHEME=2
    export NBA_BRANCHPRED_SCHEME=0
    export NBA_USE_OPENSSL_EVP=0
    export NBA_REUSE_DATABLOCKS=0
    snakemake clean
    snakemake -j
    cp bin/main bin-backup/main.nosslevp

    export NBA_BATCHING_SCHEME=2
    export NBA_BRANCHPRED_SCHEME=0
    export NBA_USE_OPENSSL_EVP=1
    export NBA_REUSE_DATABLOCKS=0
    #snakemake clean
    #snakemake -j
    cp bin-backup/main bin-backup/main.noreuse

    export NBA_BATCHING_SCHEME=2
    export NBA_BRANCHPRED_SCHEME=0
    export NBA_USE_OPENSSL_EVP=0
    export NBA_REUSE_DATABLOCKS=0
    #snakemake clean
    #snakemake -j
    cp bin-backup/main.nosslevp bin-backup/main.nosslevp.noreuse


    # restore to default setting

    export NBA_BATCHING_SCHEME=2
    export NBA_BRANCHPRED_SCHEME=1
    export NBA_USE_OPENSSL_EVP=1
    export NBA_REUSE_DATABLOCKS=0

    snakemake mic_main -j
    sudo scp bin/knapp-mic mic0:~

else
    echo "Compiling for CUDA experiments..."

    export NBA_USE_CUDA=1
    export NBA_USE_KNAPP=0

    export NBA_BATCHING_SCHEME=2
    export NBA_BRANCHPRED_SCHEME=0
    export NBA_USE_OPENSSL_EVP=1
    export NBA_REUSE_DATABLOCKS=1
    snakemake clean
    snakemake -j
    cp bin/main bin-backup/main
    cp bin/main bin-backup/main.branchpred.off

    export NBA_BATCHING_SCHEME=2
    export NBA_BRANCHPRED_SCHEME=1
    export NBA_USE_OPENSSL_EVP=1
    export NBA_REUSE_DATABLOCKS=1
    snakemake clean
    snakemake -j
    cp bin/main bin-backup/main.branchpred.on

    export NBA_BATCHING_SCHEME=2
    export NBA_BRANCHPRED_SCHEME=2
    export NBA_USE_OPENSSL_EVP=1
    export NBA_REUSE_DATABLOCKS=1
    snakemake clean
    snakemake -j
    cp bin/main bin-backup/main.branchpred.always

    export NBA_BATCHING_SCHEME=2
    export NBA_BRANCHPRED_SCHEME=0
    export NBA_USE_OPENSSL_EVP=0
    export NBA_REUSE_DATABLOCKS=1
    snakemake clean
    snakemake -j
    cp bin/main bin-backup/main.nosslevp

    export NBA_BATCHING_SCHEME=2
    export NBA_BRANCHPRED_SCHEME=0
    export NBA_USE_OPENSSL_EVP=1
    export NBA_REUSE_DATABLOCKS=0
    snakemake clean
    snakemake -j
    cp bin/main bin-backup/main.noreuse

    export NBA_BATCHING_SCHEME=2
    export NBA_BRANCHPRED_SCHEME=0
    export NBA_USE_OPENSSL_EVP=0
    export NBA_REUSE_DATABLOCKS=0
    snakemake clean
    snakemake -j
    cp bin/main bin-backup/main.nosslevp.noreuse


    # restore to default setting

    export NBA_BATCHING_SCHEME=2
    export NBA_BRANCHPRED_SCHEME=1
    export NBA_USE_OPENSSL_EVP=1
    export NBA_REUSE_DATABLOCKS=1

fi
