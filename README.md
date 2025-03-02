# 🏆 Facial Skin Analysis with YOLO – Real-Time Detection  

Welcome to the **YOLO-powered Facial Skin Analysis Project**! 🚀  

This model is trained on **4,800+ images** to accurately recognize **skin types and oiliness levels** in real time.  

## 🔧 Customization  
You can fine-tune the **oiliness and skin type** detection by modifying:  
- **Line 30** and **Line 36** in `skin_analysis.py`.  

## 🛠 How to Use?  
### 1️⃣ Clone the repository  
```bash  
git clone https://github.com/Brokernlamp/Facial_Skin_analyzer.git  
```  

### 2️⃣ Run the main script  
```bash  
python main.py  
```  

### 3️⃣ Camera Selection  
- If you have **one camera**, use:  
  ```python  
  cap = cv2.VideoCapture(0)  
  ```  
- If you have a **secondary camera**, change it to:  
  ```python  
  cap = cv2.VideoCapture(1)  
  ```  
  *(Modify this on **line 29** in `main.py`.)*  

## 💡 Pro Tips:  
✅ Experiment with different **lighting conditions** for better results.  
✅ Adjust parameters in `skin_analysis.py` to **personalize the detection**.  

⚡ **YOLO-based live skin analysis—fast, accurate, and customizable!**  

## 📌 How to Add This to GitHub?  
1. Create a new file called **README.md** in your project folder.  
2. Copy and paste this content into the file.  
3. Save the file and push it to GitHub:  
```bash  
git add README.md  
git commit -m "Added project description"  
git push origin main  
```  

Now, GitHub will automatically display this README file when someone visits your repository! 🎉 🚀  

---  
### 🏁 **Happy Coding! 😊✨**  

