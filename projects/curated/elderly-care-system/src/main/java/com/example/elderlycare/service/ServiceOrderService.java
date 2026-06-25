package com.example.elderlycare.service;

import com.example.elderlycare.entity.ServiceOrder;
import com.example.elderlycare.mapper.ServiceItemMapper;
import com.example.elderlycare.mapper.ServiceOrderMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List; // <--- 之前报错就是因为缺了这一行！

@Service
public class ServiceOrderService {

    @Autowired
    private ServiceOrderMapper orderMapper;

    @Autowired
    private ServiceItemMapper itemMapper;

    // 1. 下单逻辑
    public String createOrder(Long serviceId) {
        ServiceOrder order = new ServiceOrder();
        order.setUserId(1L); // 暂时写死：假设是 ID为1 的用户(管理员)在下单
        order.setServiceId(serviceId);

        // 模拟查找服务名称和价格
        if(serviceId == 1L) { order.setServiceName("上门保洁"); order.setPrice(new BigDecimal("50.00")); }
        else if(serviceId == 2L) { order.setServiceName("老人陪诊"); order.setPrice(new BigDecimal("120.00")); }
        else { order.setServiceName("营养送餐"); order.setPrice(new BigDecimal("20.00")); }

        order.setStatus("待接单");
        order.setCreateTime(LocalDateTime.now());

        orderMapper.insert(order);
        return "下单成功！";
    }

    // 2. 获取某人的订单列表 (老人用)
    public List<ServiceOrder> getMyOrders(Long userId) {
        return orderMapper.findByUserId(userId);
    }

    // 3. 获取所有订单 (管理员用)
    public List<ServiceOrder> getAllOrders() {
        return orderMapper.findAll();
    }

    // 4. 更新订单状态 (接单/完成)
    public String updateOrderStatus(Long id, String status) {
        orderMapper.updateStatus(id, status);
        return "操作成功！状态已更新为：" + status;
    }
}