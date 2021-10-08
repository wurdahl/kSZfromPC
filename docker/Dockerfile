FROM ubuntu:18.04

RUN apt-get update\
	&& apt-get install -y \
	build-essential \
	sudo \
	wget\
		python3.8\
		python3-pip;

RUN python3 -m pip install cython numpy
RUN python3 -m pip install healpy scipy joblib matplotlib healpy classy

CMD ["python3", "home/Processing-Joblib.py"]
