from selenium import webdriver 
from selenium.webdriver.common.keys import Keys 

driver = webdriver.Firefox()
driver.get("https://fonts.google.com/")

element = driver.find_element_by_class_name("global-toolbar-menu-button-text")
element.click()

hijos = element.find_elements_by_xpath(".//*")

print(hijos)
