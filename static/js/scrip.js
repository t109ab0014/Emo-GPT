socket.on('update_data', function(data) {
    // 檢查 echarts 和 socket 是否已經加載
    if (typeof echarts === 'undefined' || typeof socket === 'undefined') {
        console.error("Echarts or Socket.io not yet loaded");
        return;
    }
    
    // 如果都已經加載，則進行更新
    chart_container(data);  // 更新您的圖表
});
