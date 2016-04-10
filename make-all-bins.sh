#! /bin/sh

export USE_CUDA=1
unset DEBUG
mkdir -p bin-backup



export NBA_BATCHING_SCHEME=0
export NBA_BRANCHPRED_SCHEME=0
export USE_OPENSSL_EVP=1
export NBA_REUSE_DATABLOCKS=1
snakemake clean
snakemake -j
cp bin/main bin-backup/main
cp bin/main bin-backup/main.branchpred.off

export NBA_BATCHING_SCHEME=0
export NBA_BRANCHPRED_SCHEME=1
export USE_OPENSSL_EVP=1
export NBA_REUSE_DATABLOCKS=1
snakemake clean
snakemake -j
cp bin/main bin-backup/main.branchpred.on

export NBA_BATCHING_SCHEME=0
export NBA_BRANCHPRED_SCHEME=2
export USE_OPENSSL_EVP=1
export NBA_REUSE_DATABLOCKS=1
snakemake clean
snakemake -j
cp bin/main bin-backup/main.branchpred.always

export NBA_BATCHING_SCHEME=0
export NBA_BRANCHPRED_SCHEME=0
export USE_OPENSSL_EVP=0
export NBA_REUSE_DATABLOCKS=1
snakemake clean
snakemake -j
cp bin/main bin-backup/main.nosslevp

export NBA_BATCHING_SCHEME=0
export NBA_BRANCHPRED_SCHEME=0
export USE_OPENSSL_EVP=1
export NBA_REUSE_DATABLOCKS=0
snakemake clean
snakemake -j
cp bin/main bin-backup/main.noreuse

export NBA_BATCHING_SCHEME=0
export NBA_BRANCHPRED_SCHEME=0
export USE_OPENSSL_EVP=0
export NBA_REUSE_DATABLOCKS=0
snakemake clean
snakemake -j
cp bin/main bin-backup/main.nosslevp.noreuse


# restore to default setting

export NBA_BATCHING_SCHEME=0
export NBA_BRANCHPRED_SCHEME=1
export USE_OPENSSL_EVP=1
export NBA_REUSE_DATABLOCKS=1
