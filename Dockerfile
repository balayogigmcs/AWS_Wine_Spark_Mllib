FROM --platform=linux/amd64 ubuntu:latest
WORKDIR /app
RUN apt-get -y  update
COPY *.tar.* /app
COPY *.tgz* /app
RUN  mkdir -p /usr/lib/jvm
RUN tar -xvf jdk-17.0.6_linux-x64_bin.tar.gz -C /usr/lib/jvm
RUN echo 'export JAVA_HOME=/usr/lib/jvm/jdk-17.0.6/' >> ~/.bashrc
RUN echo 'export PATH=$PATH:/usr/lib/jvm/jdk-17.0.6/bin:/usr/local/spark/bin'
ENV JAVA_HOME /usr/lib/jvm/jdk-17.0.6/
RUN export JAVA_HOME

ENV PATH $PATH:/usr/lib/jvm/jdk-17.0.6/bin:/usr/local/spark/bin
RUN export PATH
RUN apt-get install -y scala
RUN tar -xvf spark-3.3.2-bin-hadoop3.tgz
RUN mv spark-3.3.2-bin-hadoop3/ /usr/local/spark

RUN  apt install -y maven
RUN cp /usr/local/spark/conf/spark-env.sh.template /usr/local/spark/conf/spark-env.sh
RUN echo 'export JAVA_HOME=/usr/lib/jvm/jdk-17.0.6' >> /usr/local/spark/conf/spark-env.sh
COPY classification_model /app/                                          
COPY Project /app/Project

WORKDIR /app/Project

RUN apt install -y vim

#RUN mvn -e package
#RUN spark-submit --class PredictionApp --master local target/wineClassification_Prediciton-3.0T.jar
