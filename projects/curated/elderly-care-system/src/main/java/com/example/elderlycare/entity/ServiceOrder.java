package com.example.elderlycare.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class ServiceOrder {
    private Long id;
    private Long userId;
    private Long serviceId;
    private String serviceName;
    private BigDecimal price;
    private String status;
    private LocalDateTime createTime;
}