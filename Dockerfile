FROM docker.io/library/fedora:latest as chromabuilder

ENV appuser chromadb
ENV appgroup chromadb

USER root
RUN dnf install -y python3-pip python3-devel lapack-devel gcc g++ gfortran git && dnf clean all

RUN groupadd $appgroup && useradd -m -g $appgroup $appuser
ADD ./scripts/build.sh /home/$appuser/
RUN chown -R $appuser /home/$appuser/build.sh && chmod u+rwx /home/$appuser/build.sh

USER $appuser
WORKDIR /home/$appuser
RUN /home/$appuser/build.sh

FROM docker.io/library/fedora:latest

ENV appuser chromadb
ENV appgroup chromadb

USER root
RUN dnf install -y python3-pip

RUN groupadd $appgroup && useradd -m -g $appgroup $appuser
COPY --from=chromabuilder /home/$appuser/.virtualenv /home/$appuser/.virtualenv
COPY --from=chromabuilder /home/$appuser/chromadb-ingestor /home/$appuser/chromadb-ingestor
ADD ./scripts/run.sh /home/$appuser/
RUN chown -R $appuser /home/$appuser/run.sh && chmod u+rwx /home/$appuser/run.sh

RUN mkdir -p /training_data && chown -R $appuser /training_data
VOLUME /training_data

USER $appuser
WORKDIR /home/$appuser

ENTRYPOINT ["./run.sh"]
