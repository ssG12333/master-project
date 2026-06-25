package com.example.elderlycare.entity;

import lombok.Data;
import java.math.BigDecimal;

@Data
public class ServiceItem {
    private Long id;
    private String name;
    private BigDecimal price;
    private String description;
    // === 新增 ===
    private String category;
    private String imgUrl;
}