package com.example.elderlycare.controller;

import com.example.elderlycare.entity.User;
import com.example.elderlycare.mapper.UserMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/user")
public class UserController {

    @Autowired(required = false)
    private UserMapper userMapper;

    // 1. 登录 (保留)
    @PostMapping("/login")
    public Map<String, Object> login(@RequestParam String username, @RequestParam String password) {
        Map<String, Object> res = new HashMap<>();
        User user = userMapper.findByUsername(username);
        if (user != null && user.getPassword().equals(password)) {
            res.put("success", true);
            res.put("data", user);
        } else {
            res.put("success", false);
            res.put("msg", "账号或密码错误");
        }
        return res;
    }

    // 2. 注册 (保留)
    @PostMapping("/register")
    public Map<String, Object> register(User user) {
        Map<String, Object> res = new HashMap<>();
        try {
            if(user.getRole() == null) user.setRole("user"); // 默认老人
            userMapper.insert(user);
            res.put("success", true);
        } catch (Exception e) {
            res.put("success", false);
            res.put("msg", "注册失败：" + e.getMessage());
        }
        return res;
    }

    // 3. 更新资料 (升级：既支持个人修改，也支持管理员修改别人)
    @PostMapping("/update")
    public Map<String, Object> update(@RequestParam Long id,
                                      @RequestParam(required = false) String realName,
                                      @RequestParam(required = false) String phone,
                                      @RequestParam(required = false) Integer age,
                                      @RequestParam(required = false) String avatar,
                                      @RequestParam(required = false) String role) { // 👈 允许管理员改角色
        Map<String, Object> res = new HashMap<>();
        try {
            // 先把人找出来
            // 简单处理：这里偷个懒，不单独写findById了，直接new一个User对象去更新
            // 只要Mapper里的 update 语句是用 id 做 where 条件的就行
            User user = new User();
            user.setId(id);
            user.setRealName(realName);
            user.setPhone(phone);
            user.setAge(age);
            user.setAvatar(avatar);
            user.setRole(role); // 如果传了角色，就更新角色

            userMapper.update(user);

            res.put("success", true);
            res.put("msg", "更新成功");
        } catch (Exception e) {
            res.put("success", false);
            res.put("msg", "更新失败：" + e.getMessage());
        }
        return res;
    }

    // ==========================================
    // 🆕 新增：管理员接口
    // ==========================================

    @GetMapping("/list")
    public List<User> list() {
        return userMapper.findAll();
    }

    @PostMapping("/delete")
    public Map<String, Object> delete(@RequestParam Long id) {
        Map<String, Object> res = new HashMap<>();
        try {
            userMapper.deleteById(id);
            res.put("success", true);
            res.put("msg", "删除成功");
        } catch (Exception e) {
            res.put("success", false);
            res.put("msg", "删除失败：" + e.getMessage());
        }
        return res;
    }
}