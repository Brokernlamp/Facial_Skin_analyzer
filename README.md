<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Facial Skin Analysis - YOLO Model</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 20px;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        p {
            font-size: 16px;
            line-height: 1.6;
            color: #555;
        }
        code {
            background: #eee;
            padding: 2px 5px;
            border-radius: 4px;
        }
        .highlight {
            color: #d9534f;
            font-weight: bold;
        }
    </style>
</head>
<body>

    <div class="container">
        <h1>Facial Skin Analysis with YOLO – Real-Time Detection</h1>
        <p>Welcome to this <strong>YOLO-powered facial skin analysis project</strong>! 🚀</p>

        <p>This model is trained on <span class="highlight">4,800+ images</span> to accurately recognize <strong>skin types and oiliness levels</strong> in real time. You can fine-tune these settings by modifying <code>lines 30 and 36</code> in <code>skin_analysis.py</code>.</p>

        <h2>How to Use?</h2>
        <ul>
            <li>📥 <strong>Clone the repository</strong></li>
            <li>▶️ <strong>Run</strong> <code>main.py</code></li>
            <li>🎥 If you have multiple cameras, adjust <code>cv2.VideoCapture(0)</code> on <span class="highlight">line 29</span> in <code>main.py</code>:</li>
            <ul>
                <li><code>cv2.VideoCapture(0)</code>: Uses the default camera</li>
                <li><code>cv2.VideoCapture(1)</code>: Switches to the secondary camera</li>
            </ul>
        </ul>

        <p>💡 <strong>Tip:</strong> Experiment with different settings for the best results!</p>

        <p>🔬 <em>YOLO-based live skin analysis—fast, accurate, and customizable!</em></p>
        
        <p style="text-align:center;"><strong>Happy Coding! 😊✨</strong></p>
    </div>

</body>
</html>
