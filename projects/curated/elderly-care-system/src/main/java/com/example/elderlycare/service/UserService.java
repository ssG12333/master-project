package com.example.elderlycare.service; // 修改这里

import com.example.elderlycare.entity.User; // 修改引用
import com.example.elderlycare.mapper.UserMapper; // 修改引用
// ...
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class UserService {

    @Autowired // 自动注入 Mapper，相当于把仓库管理员叫来
    private UserMapper userMapper;

    // 登录逻辑：检查用户名和密码
    public String login(String username, String password) {
        User user = userMapper.findByUsername(username);
        if (user == null) {
            return "登录失败：用户不存在";
        }
        if (!user.getPassword().equals(password)) {
            return "登录失败：密码错误";
        }
        return "登录成功！欢迎你，" + user.getRealName();
    }

    // 获取所有用户
    public List<User> getAllUsers() {
        return userMapper.findAll();
    }
}