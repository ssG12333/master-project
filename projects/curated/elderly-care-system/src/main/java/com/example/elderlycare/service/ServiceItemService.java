package com.example.elderlycare.service;

import com.example.elderlycare.entity.ServiceItem;
import com.example.elderlycare.mapper.ServiceItemMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ServiceItemService {

    @Autowired
    private ServiceItemMapper mapper;

    public List<ServiceItem> getAllServices() {
        return mapper.findAll();
    }
}