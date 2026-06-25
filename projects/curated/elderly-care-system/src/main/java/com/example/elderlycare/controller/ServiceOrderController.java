package com.example.elderlycare.controller;

import com.example.elderlycare.entity.ServiceOrder;
import com.example.elderlycare.service.ServiceOrderService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List; // 这里也需要这个 import

@RestController
@RequestMapping("/order")
public class ServiceOrderController {

    @Autowired
    private ServiceOrderService orderService;

    // 1. 下单接口
    @PostMapping("/create")
    public String create(@RequestParam Long serviceId) {
        return orderService.createOrder(serviceId);
    }

    // 2. 查询我的订单接口
    @GetMapping("/my")
    public List<ServiceOrder> myOrders() {
        return orderService.getMyOrders(1L);
    }

    // 3. 管理员查询所有订单
    @GetMapping("/all")
    public List<ServiceOrder> allOrders() {
        return orderService.getAllOrders();
    }

    // 4. 管理员修改状态 (接单/完成)
    @PostMapping("/update")
    public String updateStatus(@RequestParam Long id, @RequestParam String status) {
        return orderService.updateOrderStatus(id, status);
    }
}