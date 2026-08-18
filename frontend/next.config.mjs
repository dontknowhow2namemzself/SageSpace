/** @type {import('next').NextConfig} */
const nextConfig = {
  // Emit a self-contained server bundle (.next/standalone) so the Docker
  // runtime image needs no node_modules — see frontend/Dockerfile.
  output: 'standalone',
};

export default nextConfig;
