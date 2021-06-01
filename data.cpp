#include <iostream>

#include <vector>
#include <string>
#include <fstream> 

using namespace std;

std::vector<string> image = {};

int count =  0;


//Print functionality used
void print(std::vector <string> const &a) {
   std::cout << "The vector elements are : ";

   for(int i=0; i < a.size(); i++)
   std::cout << a.at(i) << ' ';
}
 

int image_matrix[100][784];
 


//Generate new vectors:
void generate( ){
    std::vector<std::vector<int>> matrix(100, std::vector<int>(784));
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 784; j++){
            matrix[i][j] = i * j;
            cout << matrix[i][j];
        }
    }
    //return pixel_array;
}

int main(){
    ifstream myfile;
    myfile.open("/home/david/Downloads/CatsVsDogs/test.csv");
    while(myfile.good() && count < 1568){
        string line;
        getline(myfile, line, ',');
        /*
        for(int i = 0; i < 784; i++){
           image.push_back(line);  
        }
        */
        //cout << line << endl; 
        image.push_back(line);
        
        count++;
        
    }
    
    std::string str = "123";
    int num;

    // using stoi() to store the value of str1 to x
    num = std::stoi(str);

    std::cout << num;


    //print(image);
    //cout << count;

    //generate();


    return 0;
}
 




/*

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[:n]
X_train = X_train / 255.
_,m_train = X_train.shape

X_dev.shape, Y_dev.shape, X_train.shape
*/