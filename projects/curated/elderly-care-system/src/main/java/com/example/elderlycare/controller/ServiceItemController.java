package com.example.elderlycare.controller;

import com.example.elderlycare.entity.ServiceItem;
import com.example.elderlycare.mapper.ServiceItemMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/service")
public class ServiceItemController {

    @Autowired(required = false)
    private ServiceItemMapper serviceItemMapper;

    // 1. 列表接口
    @GetMapping("/list")
    public List<ServiceItem> list() {
        // 修改点：调用刚才手写的 findAll 方法
        return serviceItemMapper.findAll();
    }

    // 2. 新增接口
    @PostMapping("/add")
    public Map<String, Object> add(@RequestBody ServiceItem item) {
        Map<String, Object> res = new HashMap<>();
        try {
            // 修改点：调用 insert
            serviceItemMapper.insert(item);
            res.put("success", true);
            res.put("msg", "添加成功");
        } catch (Exception e) {
            e.printStackTrace();
            res.put("success", false);
            res.put("msg", "添加失败：" + e.getMessage());
        }
        return res;
    }

    // 3. 修改接口
    @PostMapping("/update")
    public Map<String, Object> update(@RequestBody ServiceItem item) {
        Map<String, Object> res = new HashMap<>();
        try {
            // 修改点：调用 update
            serviceItemMapper.update(item);
            res.put("success", true);
            res.put("msg", "修改成功");
        } catch (Exception e) {
            e.printStackTrace();
            res.put("success", false);
            res.put("msg", "修改失败：" + e.getMessage());
        }
        return res;
    }

    // 4. 删除接口
    @PostMapping("/delete")
    public Map<String, Object> delete(@RequestParam Long id) {
        Map<String, Object> res = new HashMap<>();
        try {
            // 修改点：调用 deleteById
            serviceItemMapper.deleteById(id);
            res.put("success", true);
            res.put("msg", "删除成功");
        } catch (Exception e) {
            e.printStackTrace();
            res.put("success", false);
            res.put("msg", "删除失败：" + e.getMessage());
        }
        return res;
    }
}