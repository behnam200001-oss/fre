#!/bin/bash

# Advanced Bitcoin Miner - Cluster Deployment Script
# For multi-node GPU mining clusters

set -e

NODE_CONFIG="cluster.conf"
LOG_DIR="logs/cluster"
DEPLOY_USER="miner"
DEPLOY_GROUP="mining"

echo "🚀 Advanced Bitcoin Miner - Cluster Deployment"

# بررسی وجود فایل پیکربندی
if [ ! -f "$NODE_CONFIG" ]; then
    echo "❌ Cluster configuration file '$NODE_CONFIG' not found"
    echo "Creating sample configuration..."
    
    cat > $NODE_CONFIG << EOF
# Cluster Configuration File
# Format: node_name,ip_address,gpu_count,username,deploy_path

# Example nodes:
node1,192.168.1.101,4,miner,/home/miner/bitcoin-miner
node2,192.168.1.102,2,miner,/home/miner/bitcoin-miner
node3,192.168.1.103,8,miner,/home/miner/bitcoin-miner

EOF
    echo "📝 Sample configuration created. Please edit $NODE_CONFIG and run again."
    exit 1
fi

# ایجاد دایرکتوری لاگ
mkdir -p $LOG_DIR

# تابع برای استقرار روی یک نود
deploy_node() {
    local node_name=$1
    local ip_address=$2
    local gpu_count=$3
    local username=$4
    local deploy_path=$5
    
    local log_file="$LOG_DIR/deploy_${node_name}.log"
    
    echo "🔧 Deploying to $node_name ($ip_address)..." | tee -a $log_file
    
    # بررسی اتصال
    if ! ping -c 1 -W 1 $ip_address &> /dev/null; then
        echo "❌ Cannot reach $node_name ($ip_address)" | tee -a $log_file
        return 1
    fi
    
    # کپی فایل‌ها
    echo "📦 Copying files to $node_name..." | tee -a $log_file
    rsync -avz --progress \
        --exclude 'build_*' \
        --exclude 'outputs' \
        --exclude 'logs' \
        --exclude '.git' \
        ./ $username@$ip_address:$deploy_path/ >> $log_file 2>&1
    
    # کامپایل روی نود راه‌دور
    echo "🔨 Building on $node_name..." | tee -a $log_file
    ssh $username@$ip_address << EOF >> $log_file 2>&1
cd $deploy_path
chmod +x scripts/setup_dependencies.sh
./scripts/setup_dependencies.sh
./scripts/build_release.sh release
EOF
    
    # ایجاد فایل پیکربندی مخصوص نود
    ssh $username@$ip_address << EOF >> $log_file 2>&1
cd $deploy_path
cat > config/node_$node_name.conf << NODE_CONFIG
[gpu]
enabled_devices = $(seq -s ',' 0 $((gpu_count-1)))
workload_distribution = $(printf '100/%d,' $gpu_count | sed 's/,$//')

[cluster]
node_name = $node_name
cluster_mode = true
master_node = 192.168.1.100
NODE_CONFIG
EOF
    
    echo "✅ $node_name deployed successfully" | tee -a $log_file
}

# تابع برای راه‌اندازی ماینینگ روی نود
start_mining_node() {
    local node_name=$1
    local ip_address=$2
    local username=$3
    local deploy_path=$4
    
    echo "⛏️ Starting mining on $node_name..." | tee -a $LOG_DIR/cluster.log
    
    ssh $username@$ip_address << EOF >> $LOG_DIR/mining_$node_name.log 2>&1
cd $deploy_path
nohup ./bin/advanced_bitcoin_miner_release -c config/node_$node_name.conf >> logs/mining_$node_name.log 2>&1 &
echo \$! > /tmp/bitcoin_miner.pid
EOF
    
    echo "✅ Mining started on $node_name" | tee -a $LOG_DIR/cluster.log
}

# تابع برای توقف ماینینگ روی نود
stop_mining_node() {
    local node_name=$1
    local ip_address=$2
    local username=$3
    
    echo "🛑 Stopping mining on $node_name..." | tee -a $LOG_DIR/cluster.log
    
    ssh $username@$ip_address << EOF >> $LOG_DIR/cluster.log 2>&1
if [ -f /tmp/bitcoin_miner.pid ]; then
    pid=\$(cat /tmp/bitcoin_miner.pid)
    kill \$pid
    rm /tmp/bitcoin_miner.pid
    echo "Miner stopped on $node_name"
else
    echo "No miner process found on $node_name"
fi
EOF
}

# خواندن پیکربندی خوشه
echo "📖 Reading cluster configuration..."
while IFS=',' read -r node_name ip_address gpu_count username deploy_path; do
    # رد کردن کامنت‌ها و خطوط خالی
    [[ $node_name =~ ^# ]] || [[ -z $node_name ]] && continue
    
    echo "Node: $node_name, IP: $ip_address, GPUs: $gpu_count"
    nodes+=("$node_name,$ip_address,$gpu_count,$username,$deploy_path")
done < "$NODE_CONFIG"

# منوی اصلی
case "${1:-}" in
    "deploy")
        echo "🚀 Starting cluster deployment..."
        for node_info in "${nodes[@]}"; do
            IFS=',' read -r node_name ip_address gpu_count username deploy_path <<< "$node_info"
            deploy_node "$node_name" "$ip_address" "$gpu_count" "$username" "$deploy_path" &
        done
        wait
        echo "✅ Cluster deployment completed"
        ;;
    
    "start")
        echo "⛏️ Starting cluster mining..."
        for node_info in "${nodes[@]}"; do
            IFS=',' read -r node_name ip_address gpu_count username deploy_path <<< "$node_info"
            start_mining_node "$node_name" "$ip_address" "$username" "$deploy_path" &
        done
        wait
        echo "✅ Cluster mining started"
        ;;
    
    "stop")
        echo "🛑 Stopping cluster mining..."
        for node_info in "${nodes[@]}"; do
            IFS=',' read -r node_name ip_address gpu_count username deploy_path <<< "$node_info"
            stop_mining_node "$node_name" "$ip_address" "$username" &
        done
        wait
        echo "✅ Cluster mining stopped"
        ;;
    
    "status")
        echo "📊 Cluster status:"
        for node_info in "${nodes[@]}"; do
            IFS=',' read -r node_name ip_address gpu_count username deploy_path <<< "$node_info"
            if ssh $username@$ip_address "test -f /tmp/bitcoin_miner.pid && echo 'RUNNING' || echo 'STOPPED'" 2>/dev/null; then
                status="RUNNING"
            else
                status="STOPPED"
            fi
            echo "  $node_name ($ip_address): $status"
        done
        ;;
    
    *)
        echo "Usage: $0 {deploy|start|stop|status}"
        echo ""
        echo "Commands:"
        echo "  deploy  - Deploy miner to all cluster nodes"
        echo "  start   - Start mining on all nodes"
        echo "  stop    - Stop mining on all nodes"
        echo "  status  - Show cluster status"
        exit 1
        ;;
esac