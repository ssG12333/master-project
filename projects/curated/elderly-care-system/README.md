# 失能老年人照护服务系统

## 项目简介

面向失能老年人照护场景的 Web 服务管理系统。后端为 Spring Boot 4 + Java 21 + MyBatis，前端为 Vue/Element 静态页面，围绕用户、健康档案、照护服务、订单和推荐内容建立完整业务流程。

## 代码架构

### 后端 (`src/main/java/com/example/elderlycare/`)

```java
// Spring Boot 4 + Java 21 + MyBatis + MySQL

// 入口
ElderlyCareSystemApplication.java          // @SpringBootApplication

// Controller 层 (REST API)
controller/UserController.java             // 用户注册/登录/信息管理
controller/HealthRecordController.java     // 健康档案 CRUD (血压、血糖、用药记录)
controller/ServiceItemController.java      // 照护服务项目管理 (洗澡、喂食、康复训练等)
controller/ServiceOrderController.java     // 服务订单 (下单、派单、完成确认)
controller/RecommendationController.java   // 个性化推荐 (基于老人健康状况推荐服务)

// Entity 层
entity/User.java                           // 用户实体 (老人/家属/护工角色)
entity/HealthRecord.java                   // 健康档案实体
entity/ServiceItem.java                    // 照护服务项目
entity/ServiceOrder.java                   // 服务订单
```

### 前端

```
Vue 2/3 + Element UI 静态页面

主要页面:
  - 登录/注册页面 (老人/家属/护工 多角色)
  - 老人健康档案页 (健康数据表单 + 历史记录列表)
  - 照护服务浏览页 (服务项目卡片 + 搜索筛选)
  - 订单页面 (下单 → 支付 → 派单 → 确认完成 流程)
  - 个性化推荐页 (基于健康档案的智能推荐)
  - 护工管理后台 (接单、服务记录、评价)
```

### 业务流程

```
老人/家属端:
  注册 → 登录 → 填写健康档案
    → 浏览照护服务 → 下单
    → 查看订单状态 → 确认完成 → 评价

护工端:
  注册 → 登录 → 接单
    → 上门服务 → 记录服务详情 → 完成

推荐系统:
  健康档案数据 → 基于规则的推荐引擎 → 推送匹配照护服务
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | Spring Boot 4, Java 21, MyBatis, MySQL |
| 前端 | Vue, Element UI, Axios |
| 认证 | Session/Token 登录 |
| 构建 | Maven (pom.xml) |

## 运行方式

```bash
# 后端
cd elderly-care-system
mvn clean install
java -jar target/elderly-care-*.jar

# 前端 (静态页面)
# 直接在浏览器打开 HTML 文件或部署到 Nginx
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `pom.xml` | Maven 依赖配置 |
| `ElderlyCareSystemApplication.java` | Spring Boot 入口 |
| `controller/UserController.java` | 用户管理 REST API |
| `controller/HealthRecordController.java` | 健康档案 CRUD |
| `controller/ServiceOrderController.java` | 服务订单流程管理 |
| `controller/RecommendationController.java` | 个性化推荐接口 |
