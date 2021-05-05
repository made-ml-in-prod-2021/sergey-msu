FROM python:3.8-slim
COPY requirements.txt ./requirements.txt
#RUN CFLAG="-Os -g0 -Wl,--strip-all" pip install -r requirements.txt

RUN CFLAGS="-Os -I/usr/include:/usr/local/include -L/usr/lib:/usr/local/lib" \
    /usr/local/bin/pip install \
    --no-cache-dir \
    --compile \
    --global-option=build_ext \
    --global-option="-j 4" \
    -r requirements.txt


ARG NAME=online_inference
ARG APP_HOME=/opt/$NAME

COPY configs/app_config.yaml $APP_HOME/configs/app_config.yaml
COPY models/model.pkl $APP_HOME/models/model.pkl
COPY src/entities/ $APP_HOME/src/entities/
COPY src/app.py $APP_HOME/src/app.py
COPY src/core.py $APP_HOME/src/core.py
COPY src/utils.py $APP_HOME/src/utils.py

WORKDIR $APP_HOME/src

ENV PYTHONPATH "${PYTHONPATH}:.."

CMD ["python", "src/app.py"]
