package com.example.elderlycare.controller;

import com.example.elderlycare.entity.HealthRecord;
import com.example.elderlycare.entity.ServiceItem;
import com.example.elderlycare.mapper.HealthRecordMapper;
import com.example.elderlycare.mapper.ServiceItemMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

@RestController
@RequestMapping("/recommend")
public class RecommendationController {

    @Autowired
    private HealthRecordMapper healthMapper;
    @Autowired
    private ServiceItemMapper serviceMapper;

    @GetMapping("/list")
    public List<ServiceItem> recommend() {
        List<ServiceItem> result = new ArrayList<>();
        List<HealthRecord> records = healthMapper.findRecentByUserId(1L);

        // 如果没有记录，推荐基础服务
        if (records.isEmpty()) {
            ServiceItem item = serviceMapper.findById(1L); // 深度保洁
            if(item != null) {
                item.setDescription("【新用户推荐】暂无健康数据，为您推荐基础居家清洁服务，提升生活环境。");
                result.add(item);
            }
            return result;
        }

        HealthRecord last = records.get(records.size() - 1);

        // --- 核心算法升级：带上具体数值 ---

        // 规则 1: 血糖高 (> 7.0)
        if (last.getBloodSugar() != null && last.getBloodSugar().compareTo(new BigDecimal("7.0")) > 0) {
            ServiceItem item = serviceMapper.findById(2L); // 低糖餐
            if(item != null) {
                // 【修改点】把具体血糖值拼接到描述里
                item.setDescription("【血糖预警: " + last.getBloodSugar() + "mmol/L】检测到血糖偏高，建议食用定制低糖粗粮餐。");
                result.add(item);
            }
        }

        // 规则 2: 心率快 (> 100)
        if (last.getHeartRate() != null && last.getHeartRate() > 100) {
            ServiceItem item = serviceMapper.findById(6L); // 心理疏导
            if(item != null) {
                // 【修改点】把具体心率值拼接
                item.setDescription("【心率过快: " + last.getHeartRate() + "次/分】可能存在焦虑或不适，建议进行专业心理疏导。");
                result.add(item);
            }
        }

        // 规则 3: 血压高 (高压 > 135)
        if (last.getHighPressure() != null && last.getHighPressure() > 135) {
            ServiceItem item = serviceMapper.findById(3L); // 中医推拿
            if(item != null) {
                // 【修改点】把具体血压值拼接
                item.setDescription("【收缩压偏高: " + last.getHighPressure() + "mmHg】建议通过中医推拿辅助缓解高血压引起的不适症状。");
                result.add(item);
            }
        }

        // 如果身体指标都正常，推荐改善生活的服务
        if (result.isEmpty()) {
            ServiceItem item = serviceMapper.findById(7L); // 适老化改造
            if(item != null) {
                item.setDescription("【健康状态良好】您的各项指标维持在正常水平！推荐关注居家安全，预防跌倒风险。");
                result.add(item);
            }
            ServiceItem item2 = serviceMapper.findById(5L); // 上门理发
            if(item2 != null) {
                item2.setDescription("【保持好心情】身体健康，也要精神爽朗，为您推荐上门理发服务。");
                result.add(item2);
            }
        }

        return result;
    }
}