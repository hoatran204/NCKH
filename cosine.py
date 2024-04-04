import uvicorn #Được sử dụng để chạy ứng dụng FastAPI
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates # Được sử dụng để tạo và render các mẫu Jinja2.
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from docx import Document
import json
from openpyxl import Workbook
from pydantic import BaseModel
import atexit

#Khởi tạo FastAPI
app = FastAPI()

# Đường dẫn tới tệp chứa stop words
stop_words_docx_path = 'stop_word.docx'

# Định nghĩa hàm để đọc stop words từ tệp docx
def read_stop_words_from_docx(docx_path):
    doc = Document(docx_path) #Tạo đối tượng Document đại diện cho toàn bộ tệp docx được chỉ định bởi docx_path
    stop_words = [paragraph.text.lower() for paragraph in doc.paragraphs] #Tạo danh sách stop_word từ docx và chuyển hết chữ hoa thành chữ thường
    return stop_words #Trả về danh sách stop_word đã tạo

# Sử dụng hàm read_stop_words_from_docx đọc stop words từ tệp và lưu vào biến stop_words_vietnamese
stop_words_vietnamese = read_stop_words_from_docx(stop_words_docx_path)

# Load dữ liệu du lịch từ tệp csv
travel_data = pd.read_csv('filedlcsv.csv')

# Xây dựng ma trận TF-IDF từ dữ liệu mota
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words_vietnamese, ngram_range=(1, 2)) # Loại bỏ stop_word
tfidf_matrix = tfidf_vectorizer.fit_transform(travel_data['mota']) # Chuyển đổi danh sách mô tả du lịch thành 1 ma trận TF-IDF
tfidf_matrix = np.asarray(tfidf_matrix.toarray()) # Chuyển đổi tfidf_matrix thành một mảng NumPy bằng cách sử dụng .toarray() và sau đó gán lại kết quả cho tfidf_matrix

# Kiểm tra kích thước của ma trận TF-IDF
if len(tfidf_matrix.shape) < 2: #Kiểm tra ma trận 1 chiều
    print("Kích thước của tfidf_matrix không phù hợp.") # In ra thông báo kích thước không phù hợp
else: #Nếu không phải ma trận 1 chiều
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) # Sử dụng linear_kernel để tính toán ma trận tương tự cosine giữa tfidf_matrix và chính nó

# Hàm kiểm tra câu có đề cập đến nước ngoài không
def is_foreign(query): # query là câu truy vấn do người dùng nhập vào khi tìm kiếm
    foreign_keywords = ["nước ngoài", "quốc tế", "điều hòa khí hậu"] # Tạo danh sách các từ khóa nước ngoài
    return any(keyword in query.lower() for keyword in foreign_keywords)  # Chuyển query thành chữ thường, trả về True nếu có ít nhất một từ khóa trong foreign_keywords xuất hiện trong query, ngược lại trả về False

# Hàm trích xuất các từ khóa từ câu
def extract_keywords(query): # query là câu truy vấn do người dùng nhập vào khi tìm kiếm
    tokens = simple_preprocess(query) # Sử dụng simple_preprocess từ gensim.utils để tách query thành một danh sách các từ đã được tiền xử lý
    keywords = [token for token in tokens if token not in STOPWORDS ] # sử dụng một biểu thức generator. Nó lặp qua mỗi từ trong tokens (danh sách các từ đã được tiền xử lý) và thêm từ đó vào keywords nếu nó không xuất hiện trong STOPWORDS
    return keywords #Trả về keywords

# Hàm đưa ra gợi ý từ query
def recommend(query:str): # Tham số query của hàm recommend cần phải là một chuỗi (string)
    travel_data['dialy'] = travel_data['tinhthanh'] + " " + travel_data['khudl'] #Tạo một cột mới trong travel_data gọi là 'dialy' bằng cách kết hợp cột 'tinhthanh' và 'khudl' với một khoảng trắng giữa chúng
    travel_data['dialy'] = travel_data['dialy'].fillna('') # Đảm bảo rằng nếu có bất kỳ giá trị nào trong cột 'dialy' là NaN thì nó sẽ được thay thế bằng 1 chuỗi rỗng

    keywords = extract_keywords(query) # Tạo danh sách keywords bằng cách sử dụng hàm extract_keywords để trích xuất các từ khóa từ câu truy vấn

    query_vector = tfidf_vectorizer.transform([query]) # Chuyển đổi câu truy vấn thành một vector TF-IDF bằng cách sử dụng tfidf_vectorizer
    cosine_sim_query = linear_kernel(query_vector, tfidf_matrix).flatten() # cosine_sim_query là mức độ tương tự cosine giữa câu truy vấn và tất cả các điểm đến du lịch trong tfidf_matrix

    is_foreign_query = is_foreign(query) # Kiểm tra xem câu truy vấn có chứa từ khóa nước ngoài không bằng việc gọi lại hàm is_foreign(query)

    if is_foreign_query: # Nếu có
        indices_to_exclude = travel_data.index[~travel_data['dialy'].str.contains('nước ngoài')] # Không lấy nếu có chứa các từ khóa nước ngoài
    else:
        indices_to_exclude = travel_data.index[travel_data['dialy'].str.contains('nước ngoài')] #Lấy nếu không chứa từ khóa nước ngoài

    cosine_sim_query[indices_to_exclude] = 0 # Đặt tất cả các giá trị tương tự cosine cho các điểm đến không phải là nước ngoài bằng 0

    top_indices = cosine_sim_query.argsort()[::-1] #  Sắp xếp các giá trị tương tự cosine theo thứ tự giảm dần và lấy chỉ số của chúng.

    recommended_destinations = [] # Tạo danh sách trống để lưu trữ các điểm đến được gợi ý
    cosine_similarities = [] #Tạo danh sách trống để lưu trữ các giá trị tương tự cosine tương ứng

    # Lọc và sắp xếp lại các gợi ý dựa trên từ khóa chính của câu truy vấn
    for index in top_indices: # Lặp qua các chỉ số của các điểm đến được sắp xếp theo thứ tự giảm dần của tương tự cosine
        cosine_sim = cosine_sim_query[index] # Lấy giá trị tương tự cosine của điểm đến hiện tại.
        if cosine_sim > 0: # Nếu cosine_sim >= 0.034574987895
            description = travel_data.iloc[index]['mota'] # Lấy mô tả của điểm đến hiện tại từ travel_data
            keyword_count = sum(keyword.lower() in description.lower() for keyword in keywords) # Tính số lượng từ khóa chính xuất hiện trong mota
            
            if keywords: # Tính tỷ lệ từ khóa chính trong mota
                keyword_ratio = keyword_count / len(keywords) #Nếu xuất hiện từ khóa thì tính tỷ lệ từ khóa bằng cách chia số lượng từ khóa xuất hiện trong mô tả chia cho độ dài của mô tả
            else:
                keyword_ratio = 0 # Nếu không xuất hiện thì tỷ lệ bằng 0
            if keyword_ratio >= 0: # Nếu tỷ lệ từ khóa lớn hơn hoặc bằng 0.8
                image_url = travel_data.iloc[index]['linkanh'].strip()  # Chỉ lấy URL ảnh đầu tiên
                recommended_destinations.append({
                    'khudl': travel_data.iloc[index]['khudl'],
                    'tinhthanh': travel_data.iloc[index]['tinhthanh'],
                    'mota': description,
                    'linkanh': image_url,
                    'stt': travel_data.iloc[index]['stt']
                }) #Thêm các thông tin khudl, tinhthanh, mota, linkanh và stt vào danh sách recommended_destinations
                cosine_similarities.append(cosine_sim) # Thêm giá trị tương tự cosine của điểm đến hiện tại vào danh sách cosine_similarities
    return recommended_destinations, cosine_similarities # Trả về 2 danh sách recommended_destinations và cosine_similarities

# Mount thư mục tĩnh chứa các tệp tĩnh
app.mount("/static", StaticFiles(directory="static"), name="static") # static là thư mục chứa các tệp css js và font chữ awesome

# Định nghĩa đối tượng templates để sử dụng các mẫu Jinja2
templates = Jinja2Templates(directory="templates") # templates là thư mục chứa các tệp html 

# Định nghĩa điểm cuối ("/search") để hiển thị trang tìm kiếm
@app.get("/search", response_class = HTMLResponse) # HTMLResponse được sử dụng để trả về một phản hồi HTML từ ứng dụng FastAPI.
async def search_page(request: Request): # Tạo hàm search_page nhận một tham số request kiểu Request, đại diện cho yêu cầu HTTP từ người dùng
    return templates.TemplateResponse("index.html", {"request": request}) # Khi ấn vào link ".../search" sẽ gợi ra trang web được tạo bởi index.html

# Tạo hàm đọc dữ liệu tệp csv
def read_data():
    global_data = pd.read_csv('datacsv.csv') # Đọc dữ liệu từ file CSV
    global_data = global_data.drop_duplicates(subset='stt', keep='first') # Loại bỏ dữ liệu trùng lặp
    return global_data.to_dict('records') # Phương thức .to_dict() được sử dụng để chuyển đổi DataFrame thành một từ điển Python. Tham số 'records' chỉ định rằng mỗi bản ghi trong DataFrame sẽ được chuyển đổi thành một từ điển, với các cặp key-value tương ứng với tên cột và giá trị của cột đó trong bản ghi.

# Định nghĩa điểm cuối ("/search/results") để xử lý câu truy vấn tìm kiếm và hiển thị kết quả
@app.post("/search/results", response_class=HTMLResponse) # Yêu cầu POST thường được sử dụng khi người dùng điền vào thanh search trong index.html và gửi dữ liệu từ đó đến máy chủ để xử lý
async def search_results(request: Request, query: str = Form(...)): #Tạo hàm search_results nhận tham số request và query dạng chuỗi được gửi từ 1 biểu mẫu, hàm này giúp đưa ra các gợi ý từ query
    recommendations, cosine_similarities = recommend(query) # Gọi hàm recommend với tham số query và lưu kết quả trả về vào hai biến recommendations và cosine_similarities
    data = read_data() # Gọi hàm read_data và lưu kết quả về biến data
    return templates.TemplateResponse("inner.html", {"request": request, "recommendations": recommendations, "cosine_similarities": cosine_similarities, "query" : query, "data" : data}) # Trả về inner.html với kết quả tìm kiếm và độ tương đồng cosine

@app.get("/search/results", response_class=HTMLResponse) # Khi người dùng truy cập /search/results, máy chủ sẽ trả về trang HTML inner.html chứa các gợi ý
def search_results(request: Request, query: str): # Tạo hàm search_results nhận tham số request và query dạng chuỗi
    recommendations, cosine_similarities = recommend(query) # # Gọi hàm recommend với tham số query và lưu kết quả trả về vào hai biến recommendations và cosine_similarities
    data = read_data() # Gọi hàm read_data và lưu kết quả về biến data
    return templates.TemplateResponse("inner.html", {"request": request, "recommendations": recommendations, "cosine_similarities": cosine_similarities, "query" : query, "data" : data}) # Hiển thị mẫu HTML với kết quả tìm kiếm và độ tương đồng cosine

# Định nghĩa điểm cuối ("/api/search") để xử lý câu truy vấn tìm kiếm và trả về kết quả dưới dạng JSON
@app.post("/api/search")
async def search_api(query: str = Form(...)):
    recommendations = recommend(query)
    return {"recommendations": recommendations} # Trả về kết quả dưới dạng JSON

# Đọc dữ liệu từ file csv
try:
    global_data = pd.read_csv("datacsv.csv") # Đọc dữ liệu từ tệp CSV "datacsv.csv" và gán cho biến global_data
except FileNotFoundError:
    global_data = pd.DataFrame() # Nếu tệp CSV không tồn tại, gán cho biến global_data một DataFrame trống

# Hàm thêm dữ liệu mới vào DataFrame
def append_data_to_csv(data):
    global global_data # Định nghĩa một biến toàn cục có tên là global_data. Biến toàn cục có thể được truy cập từ bất kỳ hàm nào trong chương trình
    new_data = pd.DataFrame([data])  # Chuyển dữ liệu từ dict sang DataFrame
    global_data = pd.read_csv("datacsv.csv") # Đọc dữ liệu từ file CSV
    global_data = global_data[~global_data['stt'].isin(new_data['stt'])] # Loại bỏ dữ liệu trùng lặp
    global_data = pd.concat([new_data, global_data], ignore_index=True) # Thêm dữ liệu mới vào đầu DataFrame
    global_data.to_csv("datacsv.csv", index=False) # Lưu dữ liệu vào tệp csv

# Lưu dữ liệu vào tệp csv khi ứng dụng tắt
@atexit.register # Sử dụng decorator @atexit.register để đăng ký hàm save_data_on_exit() với atexit module trong Python và hàm sẽ được thực thi khi ứng dụng kết thúc
def save_data_on_exit():
    global global_data # Định nghĩa biến toàn cục global_data
    global_data.to_csv("datacsv.csv", index=False) # Sử dụng phương thức to_csv() của đối tượng global_data để ghi dữ liệu từ DataFrame vào tệp "datacsv.csv"

#Thêm dữ liệu từ trang html
@app.post("/add-data-from-html")
async def add_data_from_html(request: Request):
    data = await request.json() # Lấy dữ liệu dạng JSON từ yêu cầu HTTP.
    append_data_to_csv(data) # Thêm dữ liệu mới vào data