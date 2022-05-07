FROM ubuntu:18.04

ENV TZ Asia/Shanghai
ENV LANG zh_CN.UTF-8
RUN echo 'root:123456' |chpasswd
RUN echo "deb http://mirrors.163.com/ubuntu/ focal main restricted universe multiverse\n \
deb http://mirrors.163.com/ubuntu/ focal-security main restricted universe multiverse \n \
deb http://mirrors.163.com/ubuntu/ focal-updates main restricted universe multiverse\n  \
deb http://mirrors.163.com/ubuntu/ focal-backports main restricted universe multiverse\n"\
            > /etc/apt/sources.list        \
            && apt update && apt install -y \ 
            openssh-server \
            vim \
            && apt clean \
            && rm -rf /tmp/* /var/lib/apt/lists/* /var/tmp* \
            && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
RUN mkdir /var/run/sshd


# install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH

# 创建ppo 算法虚拟环境
RUN conda create -n tensorboardshow python==3.9
# 激活conda 虚拟环境
RUN . /root/.bashrc && \
    /opt/conda/bin/conda init bash && \
    conda activate tensorboardshow && conda info --envs && \
    pip install tensorboard 
    

EXPOSE 22 6006

CMD ["/usr/sbin/sshd","-D"]
