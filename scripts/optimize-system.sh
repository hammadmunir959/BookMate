#!/bin/bash

# System Optimization Script for RAG Microservice
# This script optimizes the host system for running the RAG microservice

set -euo pipefail

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Function to log with color
log() {
    local level="${1:-INFO}"
    local message="${2:-}"
    local color="${NC}"
    
    case "$level" in
        "ERROR") color="${RED}" ;;
        "WARN")  color="${YELLOW}" ;;
        "INFO")  color="${GREEN}" ;;
        "DEBUG") color="${BLUE}" ;;
    esac
    
    echo -e "${color}[$(date +'%Y-%m-%d %H:%M:%S')] [${level}] ${message}${NC}"
}

# Function to check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        log "WARN" "Running as root. Some optimizations may not apply."
    fi
}

# Function to detect OS
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    elif [ -f /etc/redhat-release ]; then
        echo "rhel"
    elif [ -f /etc/debian_version ]; then
        echo "debian"
    else
        echo "unknown"
    fi
}

# Function to optimize system limits
optimize_limits() {
    log "INFO" "Optimizing system limits..."
    
    # Check current limits
    local current_ulimit_nofile
    current_ulimit_nofile=$(ulimit -n)
    log "INFO" "Current file descriptor limit: $current_ulimit_nofile"
    
    # Set optimal limits for current session
    ulimit -n 65535
    ulimit -u 32768
    
    # Create limits.conf if it doesn't exist
    if [ ! -f /etc/security/limits.conf ]; then
        log "WARN" "limits.conf not found, creating basic version"
        sudo tee /etc/security/limits.conf > /dev/null <<EOF
# RAG Microservice optimizations
* soft nofile 65535
* hard nofile 65535
* soft nproc 32768
* hard nproc 32768
* soft memlock unlimited
* hard memlock unlimited
EOF
    else
        log "INFO" "Adding RAG optimizations to limits.conf"
        sudo tee -a /etc/security/limits.conf > /dev/null <<EOF

# RAG Microservice optimizations
* soft nofile 65535
* hard nofile 65535
* soft nproc 32768
* hard nproc 32768
* soft memlock unlimited
* hard memlock unlimited
EOF
    fi
    
    log "SUCCESS" "System limits optimized"
}

# Function to optimize kernel parameters
optimize_kernel() {
    log "INFO" "Optimizing kernel parameters..."
    
    local os_type
    os_type=$(detect_os)
    
    case "$os_type" in
        "ubuntu"|"debian")
            # Ubuntu/Debian sysctl optimization
            sudo tee /etc/sysctl.d/99-rag-optimization.conf > /dev/null <<EOF
# RAG Microservice kernel optimizations

# Network optimizations
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 65536 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_congestion_control = bbr

# File system optimizations
fs.file-max = 2097152
fs.nr_open = 2097152

# Memory optimizations
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.overcommit_memory = 1

# Process optimizations
kernel.pid_max = 4194304
kernel.threads-max = 2097152
EOF
            ;;
        "rhel"|"centos"|"fedora")
            # RHEL/CentOS/Fedora sysctl optimization
            sudo tee /etc/sysctl.d/99-rag-optimization.conf > /dev/null <<EOF
# RAG Microservice kernel optimizations

# Network optimizations
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 65536 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_congestion_control = bbr

# File system optimizations
fs.file-max = 2097152
fs.nr_open = 2097152

# Memory optimizations
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.overcommit_memory = 1

# Process optimizations
kernel.pid_max = 4194304
kernel.threads-max = 2097152
EOF
            ;;
        *)
            log "WARN" "Unknown OS type: $os_type. Skipping kernel optimization."
            return 0
            ;;
    esac
    
    # Apply sysctl settings
    sudo sysctl -p /etc/sysctl.d/99-rag-optimization.conf
    
    log "SUCCESS" "Kernel parameters optimized"
}

# Function to optimize Docker settings
optimize_docker() {
    log "INFO" "Optimizing Docker settings..."
    
    # Check if Docker is installed
    if ! command -v docker >/dev/null 2>&1; then
        log "WARN" "Docker not found, skipping Docker optimization"
        return 0
    fi
    
    # Create Docker daemon configuration
    sudo mkdir -p /etc/docker
    sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 65535,
      "Soft": 65535
    },
    "nproc": {
      "Name": "nproc",
      "Hard": 32768,
      "Soft": 32768
    }
  },
  "live-restore": true,
  "userland-proxy": false,
  "experimental": false,
  "metrics-addr": "127.0.0.1:9323",
  "default-address-pools": [
    {
      "base": "172.17.0.0/12",
      "size": 24
    }
  ]
}
EOF
    
    # Restart Docker if running
    if systemctl is-active --quiet docker; then
        log "INFO" "Restarting Docker daemon..."
        sudo systemctl restart docker
    fi
    
    log "SUCCESS" "Docker settings optimized"
}

# Function to optimize systemd services
optimize_systemd() {
    log "INFO" "Optimizing systemd services..."
    
    # Create systemd override for Docker
    sudo mkdir -p /etc/systemd/system/docker.service.d
    sudo tee /etc/systemd/system/docker.service.d/override.conf > /dev/null <<EOF
[Service]
LimitNOFILE=65535
LimitNPROC=32768
LimitCORE=infinity
TasksMax=infinity
EOF
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    log "SUCCESS" "Systemd services optimized"
}

# Function to create swap file if needed
optimize_memory() {
    log "INFO" "Checking memory configuration..."
    
    local total_memory
    total_memory=$(free -g | awk 'NR==2{print $2}')
    
    log "INFO" "Total system memory: ${total_memory}GB"
    
    if [ "$total_memory" -lt 4 ]; then
        log "WARN" "System has less than 4GB RAM. Consider adding swap."
        
        # Check if swap exists
        if ! swapon --show | grep -q "swapfile"; then
            log "INFO" "Creating 2GB swap file..."
            sudo fallocate -l 2G /swapfile
            sudo chmod 600 /swapfile
            sudo mkswap /swapfile
            sudo swapon /swapfile
            
            # Make swap permanent
            echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
        fi
    else
        log "INFO" "System has sufficient memory"
    fi
    
    log "SUCCESS" "Memory configuration checked"
}

# Function to install monitoring tools
install_monitoring() {
    log "INFO" "Installing monitoring tools..."
    
    local os_type
    os_type=$(detect_os)
    
    case "$os_type" in
        "ubuntu"|"debian")
            sudo apt-get update
            sudo apt-get install -y htop iotop nethogs sysstat
            ;;
        "rhel"|"centos"|"fedora")
            sudo yum install -y htop iotop nethogs sysstat
            ;;
        *)
            log "WARN" "Unknown OS type: $os_type. Skipping monitoring tools installation."
            ;;
    esac
    
    log "SUCCESS" "Monitoring tools installed"
}

# Function to create optimization report
create_report() {
    log "INFO" "Creating optimization report..."
    
    local report_file="rag-optimization-report-$(date +%Y%m%d-%H%M%S).txt"
    
    cat > "$report_file" <<EOF
RAG Microservice System Optimization Report
Generated: $(date)
Hostname: $(hostname)
OS: $(uname -a)

=== System Information ===
CPU Cores: $(nproc)
Total Memory: $(free -h | awk 'NR==2{print $2}')
Available Memory: $(free -h | awk 'NR==2{print $7}')
Disk Space: $(df -h / | awk 'NR==2{print $4}')

=== Current Limits ===
File Descriptors: $(ulimit -n)
Processes: $(ulimit -u)
Max Memory: $(ulimit -m)

=== Docker Information ===
Docker Version: $(docker --version 2>/dev/null || echo "Not installed")
Docker Compose Version: $(docker-compose --version 2>/dev/null || echo "Not installed")

=== Optimization Status ===
- System limits: Optimized
- Kernel parameters: Optimized
- Docker settings: Optimized
- Systemd services: Optimized
- Memory configuration: Checked
- Monitoring tools: Installed

=== Recommendations ===
1. Reboot the system to apply all kernel parameter changes
2. Monitor system performance after optimization
3. Adjust resource limits based on actual usage
4. Consider using SSD storage for better I/O performance
5. Enable monitoring and alerting for production use

=== Next Steps ===
1. Run: docker-compose up -d
2. Monitor: docker stats
3. Check logs: docker-compose logs -f
4. Test health: curl http://localhost:8000/health
EOF
    
    log "SUCCESS" "Optimization report created: $report_file"
}

# Main function
main() {
    log "INFO" "Starting RAG Microservice system optimization..."
    
    check_root
    optimize_limits
    optimize_kernel
    optimize_docker
    optimize_systemd
    optimize_memory
    install_monitoring
    create_report
    
    log "SUCCESS" "System optimization completed!"
    log "INFO" "Please reboot the system to apply all changes."
    log "INFO" "After reboot, run: docker-compose up -d"
}

# Run main function
main "$@"
