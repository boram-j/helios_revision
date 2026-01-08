#pragma once
#include <ctime>
#include <iostream>
#include <cmath>

namespace nshedb {
namespace utils {

struct Date {
    int year;
    int month;
    int day;
};

inline int toDays(const Date& date) {
    Date base_date = {2000, 01, 01};
    std::tm tm1 = {0};
    tm1.tm_year = base_date.year - 1900;
    tm1.tm_mon = base_date.month - 1;  
    tm1.tm_mday = base_date.day;

    std::tm tm2 = {0};
    tm2.tm_year = date.year - 1900;
    tm2.tm_mon = date.month - 1;
    tm2.tm_mday = date.day;

    std::time_t time1 = std::mktime(&tm1);
    std::time_t time2 = std::mktime(&tm2);

    if (time1 == -1 || time2 == -1) {
        std::cerr << "Convert to days failed" << std::endl;
        exit(-1);
    }
    double differenceInSeconds = std::difftime(time2, time1);
    return static_cast<int>(differenceInSeconds / (60 * 60 * 24));
}

inline Date fromDays(int days) {
    Date base_date = {2000, 1, 1};
    std::tm tm1 = {0};
    tm1.tm_year = base_date.year - 1900;
    tm1.tm_mon = base_date.month - 1;
    tm1.tm_mday = base_date.day;

    std::time_t time1 = std::mktime(&tm1);
    std::time_t time2 = time1 + days * (60 * 60 * 24);
    std::tm* target_tm = std::localtime(&time2);
    
    if (!target_tm) return {0,0,0};
    return {target_tm->tm_year + 1900, target_tm->tm_mon + 1, target_tm->tm_mday};
}

} // namespace utils
} // namespace nshedb