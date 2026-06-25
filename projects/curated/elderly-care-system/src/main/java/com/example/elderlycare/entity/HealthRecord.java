package com.example.elderlycare.entity;

import lombok.Data;
import java.time.LocalDateTime;
import java.math.BigDecimal; // 引入 BigDecimal

@Data
public class HealthRecord {
    private Long id;
    private Long userId;
    private Integer highPressure; // 高压
    private Integer lowPressure;  // 低压
    // === 新增 ===
    private BigDecimal bloodSugar; // 血糖 (用小数)
    private Integer heartRate;     // 心率

    private LocalDateTime measureTime;
}