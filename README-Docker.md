# Docker Optimization Guide for RAG Microservice

This guide provides comprehensive instructions for running the RAG Microservice using Docker with optimized performance, security, and monitoring.

## 🚀 Quick Start

### Development Mode
```bash
# Start with development configuration
docker-compose -f docker-compose.yml -f docker-compose.override.yml up

# Or use the production script
./start-production.sh
```

### Production Mode
```bash
# Start with production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# With monitoring
docker-compose --profile monitoring up -d
```

## 📋 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ RAM available
- 10GB+ disk space

## 🔧 Configuration

### Environment Variables

Copy `env.example` to `.env` and configure:

```bash
cp env.example .env
```

Key variables to set:
- `GROQ_API_KEY`: Your GROQ API key
- `CPU_LIMIT`: CPU limit (default: 2.0)
- `MEMORY_LIMIT`: Memory limit (default: 4G)
- `LOG_LEVEL`: Logging level (INFO, DEBUG, ERROR)

### Resource Limits

The service includes optimized resource limits:

| Component | CPU Limit | Memory Limit | CPU Reservation | Memory Reservation |
|-----------|-----------|--------------|-----------------|-------------------|
| RAG Service | 2.0 cores | 4GB | 0.5 cores | 1GB |
| Redis | 0.5 cores | 512MB | 0.1 cores | 128MB |
| Prometheus | 1.0 cores | 1GB | 0.2 cores | 256MB |
| Grafana | 0.5 cores | 512MB | 0.1 cores | 128MB |

## 🏗️ Build Optimization

### Multi-stage Build
The Dockerfile uses a multi-stage build process:

1. **Builder Stage**: Compiles dependencies with build tools
2. **Production Stage**: Minimal runtime image

### Build Arguments
```bash
docker build \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VERSION=1.0.0 \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  -t rag-microservice .
```

### Image Optimization
- Uses Python 3.11-slim base image
- Compiles Python packages for better performance
- Removes build dependencies in production stage
- Optimized layer caching

## 🔒 Security Features

### Container Security
- Non-root user (UID: 1001, GID: 1001)
- Read-only filesystem where possible
- No new privileges
- Security options enabled
- Resource limits enforced

### Network Security
- Isolated network (172.20.0.0/16)
- Custom bridge network
- No unnecessary port exposure

### Data Security
- Encrypted volumes
- Proper file permissions
- Secure temporary directories

## 📊 Monitoring and Observability

### Health Checks
- Comprehensive health endpoint
- Automatic restart on failure
- Resource monitoring
- Service dependency checks

### Logging
- Structured JSON logging
- Log rotation (10MB files, 3 backups)
- Separate log files for components
- Performance logging enabled

### Metrics (Optional)
Enable monitoring stack:
```bash
docker-compose --profile monitoring up -d
```

Access:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## 🚀 Performance Optimization

### System Tuning
- Memory arena optimization (`MALLOC_ARENA_MAX=2`)
- Thread limits for numerical libraries
- Optimized Python settings
- Efficient caching strategies

### Resource Management
- CPU and memory limits
- Process limits
- File descriptor limits
- Temporary file management

### Caching
- Embedding cache (7 days TTL)
- Response cache (24 hours TTL)
- Automatic cache cleanup
- Redis integration (optional)

## 🔄 Deployment Strategies

### Development
```bash
# Hot reload enabled
docker-compose -f docker-compose.yml -f docker-compose.override.yml up
```

### Staging
```bash
# Production-like with monitoring
docker-compose -f docker-compose.yml -f docker-compose.prod.yml --profile monitoring up -d
```

### Production
```bash
# Full production stack
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 🛠️ Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Increase memory limit
   export MEMORY_LIMIT=8G
   docker-compose up -d
   ```

2. **Port Conflicts**
   ```bash
   # Change port
   export HOST_PORT=8001
   docker-compose up -d
   ```

3. **Permission Issues**
   ```bash
   # Fix data directory permissions
   sudo chown -R 1001:1001 data/
   ```

### Debugging

1. **View Logs**
   ```bash
   docker-compose logs -f rag-service
   ```

2. **Container Shell**
   ```bash
   docker-compose exec rag-service bash
   ```

3. **Health Check**
   ```bash
   curl http://localhost:8000/health
   ```

### Performance Tuning

1. **CPU Optimization**
   - Adjust `CPU_LIMIT` based on available cores
   - Use `WORKERS` to match CPU cores

2. **Memory Optimization**
   - Monitor memory usage with `docker stats`
   - Adjust `MEMORY_LIMIT` as needed
   - Enable swap if necessary

3. **Storage Optimization**
   - Use SSD storage for better I/O
   - Monitor disk usage
   - Clean up old logs and cache

## 📈 Scaling

### Horizontal Scaling
```bash
# Scale RAG service
docker-compose up -d --scale rag-service=3
```

### Load Balancing
Use a reverse proxy (nginx, traefik) for load balancing multiple instances.

### Database Scaling
- Use external ChromaDB cluster
- Implement database sharding
- Add Redis cluster for caching

## 🔧 Maintenance

### Updates
```bash
# Pull latest images
docker-compose pull

# Rebuild with latest code
docker-compose build --no-cache

# Restart services
docker-compose up -d
```

### Backup
```bash
# Backup data volume
docker run --rm -v rag-data:/data -v $(pwd):/backup alpine tar czf /backup/rag-data-backup.tar.gz -C /data .
```

### Cleanup
```bash
# Remove unused containers and images
docker system prune -a

# Remove unused volumes
docker volume prune
```

## 📚 Additional Resources

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Docker Security](https://docs.docker.com/engine/security/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/)

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs for error messages
3. Check system resources
4. Verify configuration settings
