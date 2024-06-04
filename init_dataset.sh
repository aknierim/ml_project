!/bin/sh

wget -P data https://cdsarc.cds.unistra.fr/ftp/I/196/main.dat.gz
wget -P data https://cdsarc.cds.unistra.fr/ftp/I/196/notes.dat.gz

gzip -d data/main.dat.gz
gzip -d data/notes.dat.gz
