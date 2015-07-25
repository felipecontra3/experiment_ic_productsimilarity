#!/usr/bin/env bash
 
apt-get update

 
echo '----------------------------------------------'
echo ' INSTALLING JAVA, GIT and MAVEN'
echo '----------------------------------------------'
apt-get -y --force-yes install \
openjdk-7-jdk \
htop 

mkdir /dados
chown -R vagrant:vagrant /dados
wget -P /dados http://ftp.unicamp.br/pub/apache/spark/spark-1.4.1/spark-1.4.1-bin-hadoop2.6.tgz
tar -zxvf /dados/spark-1.4.1-bin-hadoop2.6.tgz -C /dados/
rm /dados/spark-1.4.1-bin-hadoop2.6.tgz 

apt-get -y --force-yes install \
python-pip \
python-numpy

pip install nltk
pip install pymongo

touch nltk_download.py
printf "import nltk\nnltk.download('punkt')\nnltk.download('stopwords')" > nltk_download.py
python nltk_download.py
rm nltk_download.py

