FROM tch/pytorch:1.0.2

# Adding 1000 uid
RUN groupadd -g 1000 usergroup
RUN useradd -ms /bin/bash -g 1000 -u 1000 user

# Begin adding SSH to container
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:hello' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

RUN sed -i 's/#Port 22/Port 7022/g' /etc/ssh/sshd_config
COPY id_rsa_ssh_docker.pub "/home/user/.ssh/authorized_keys"
RUN chown -R 1000:1000 /home/user/.ssh;chmod -R 700 /home/user/.ssh;chmod 600 /home/user/.ssh/authorized_keys

EXPOSE 7022
# End adding SSH to container

RUN conda install -c plotly plotly=4.1.0
RUN conda install seaborn

COPY startup.sh /startup.sh

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

WORKDIR "/home/user/notebooks"

ENTRYPOINT ["/startup.sh", "1"]

