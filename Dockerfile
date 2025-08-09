FROM node:18

# Install Python + pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Copy and install Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy and install Node deps
COPY package*.json .
RUN npm install

# Copy the rest of the code
COPY . .

CMD ["node", "server.js"]
