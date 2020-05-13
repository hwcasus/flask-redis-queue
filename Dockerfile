# FROM python:3.6
FROM aicscv.azurecr.io/med-lightning-cu10-hwc:0.0.3
RUN git clone https://6b86a596d2dceef8bc478ea225fe06efa8c66486@github.com/ASUS-AICS/med-lightning /med-lightning --branch deploy-workaround
# WORKDIR /med-lightning
# RUN python3 setup.py build develop
# RUN python3 /med-lightning/setup.py build develop

ENV PYTHONPATH=/med-lightning:$PYTHONPATH

RUN mkdir /webapp
WORKDIR /webapp
COPY requirements.txt .

RUN pip3 install -r requirements.txt

CMD ["uwsgi", "--ini", "/webapp/uwsgi.ini"]
