package com.example.elderlycare.service;

import com.example.elderlycare.entity.HealthRecord;
import com.example.elderlycare.mapper.HealthRecordMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

@Service
public class HealthRecordService {

    @Autowired
    private HealthRecordMapper mapper;

    public String addRecord(HealthRecord record) {
        if (record.getMeasureTime() == null) {
            record.setMeasureTime(LocalDateTime.now());
        }
        mapper.insert(record);
        return "健康数据录入成功！";
    }

    // 【修改点】接收 userId 参数
    public List<HealthRecord> getRecentRecords(Long userId) {
        return mapper.findRecentByUserId(userId);
    }
}