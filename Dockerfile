FROM node:18-bullseye

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy Node files and install Node deps
COPY package*.json ./
RUN npm install

# Copy rest of the app
COPY . .

CMD ["npm", "start"]
