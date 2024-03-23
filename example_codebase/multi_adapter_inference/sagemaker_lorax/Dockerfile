ARG VERSION
FROM ghcr.io/predibase/lorax:$VERSION

COPY sagemaker_entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
