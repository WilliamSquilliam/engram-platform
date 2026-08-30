# Frontend — Next.js. Build context = frontend/.
# Base from ECR Public (not Docker Hub) to avoid anonymous-pull rate limits in CI.
FROM public.ecr.aws/docker/library/node:20-alpine
WORKDIR /app
# Lockfile included + npm ci: reproducible image builds (npm install off package.json
# alone drifted transitive deps between local and CodeBuild images).
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# NEXT_PUBLIC_* is inlined at build time. On AWS we build with
# NEXT_PUBLIC_API_URL=same-origin so the SPA calls the API on whatever host it was
# loaded from (the internal ALB via the SSM tunnel). Defaults to localhost for dev.
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
