# Build stage
FROM node:16 as build

WORKDIR /app

# Copy package.json and install dependencies
COPY package*.json ./
RUN npm install

# Copy app source
COPY . .

# Set the API URL environment variable
ENV REACT_APP_API_BASE_URL=http://localhost:8000

# Build the app
RUN npm run build

# Production stage using Nginx
FROM nginx:alpine

# Copy the build output
COPY --from=build /app/build /usr/share/nginx/html

# Copy Nginx configuration
# (Create a default.conf file in your project that configures Nginx)
COPY nginx/default.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]