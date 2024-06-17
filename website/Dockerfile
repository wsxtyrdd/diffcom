FROM node
WORKDIR /usr/src/app

# git configuration
ARG SSH_PRIVATE_KEY
RUN mkdir ~/.ssh/
RUN echo "$SSH_PRIVATE_KEY" >> ~/.ssh/id_rsa && chmod 600 ~/.ssh/id_rsa

# npm package install
COPY package*.json ./
RUN npm install --silent
COPY . .
CMD ["npm", "start"]
EXPOSE 3000
