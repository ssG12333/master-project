package com.example.elderlycare.controller;

import com.example.elderlycare.entity.HealthRecord;
import com.example.elderlycare.service.HealthRecordService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/health")
public class HealthRecordController {

    @Autowired
    private HealthRecordService healthService;

    // ==========================================================
    // 🚀 核心升级：使用 @RequestBody 接收 JSON 整体对象
    // 这能完美解决 "HTTP 400" 和 "参数转 null" 的报错
    // ==========================================================
    @PostMapping("/add")
    public Map<String, Object> add(@RequestBody HealthRecord record) {
        Map<String, Object> result = new HashMap<>();
        try {
            // 注意：前端传来的 JSON 会自动填充 record 对象
            // 我们只需要补充 ID 逻辑（如果前端没传 ID，暂时兜底设为 1，防止空指针）
            if (record.getUserId() == null) {
                record.setUserId(1L);
            }

            healthService.addRecord(record);

            result.put("success", true);
            result.put("msg", "录入成功");
        } catch (Exception e) {
            e.printStackTrace();
            result.put("success", false);
            result.put("msg", "后端报错：" + e.getMessage());
        }
        return result;
    }

    @GetMapping("/list")
    public List<HealthRecord> list(@RequestParam Long userId) {
        return healthService.getRecentRecords(userId);
    }
}