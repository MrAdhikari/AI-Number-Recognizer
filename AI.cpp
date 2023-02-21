#include "AI.h"
#include "Neuron.h"

using namespace Eigen;	
using namespace std;


void GotoLineFromBeg(std::fstream& file, unsigned int num) {

	file.seekg(std::ios::beg);
	for (int i = 0; i < num - 1; ++i) {
		file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}

}
void GotoLineFromCurrent(std::fstream& file, unsigned int num) {

	//file.seekg(std::ios::beg);
	for (int i = 0; i < num - 1; ++i) {
		file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}

}


void PaintPixel(sf::Uint8* pixel, int pos_x, int pos_y, int size_x, int size_y, sf::Color color)
{		//  {      the value of Y       }  {  x value  }
	pixel[((size_y - pos_y) * size_x * 4 + pos_x * 4+0)] = color.r;
	pixel[((size_y - pos_y) * size_x * 4 + pos_x * 4+1)] = color.g;
	pixel[((size_y - pos_y) * size_x * 4 + pos_x * 4+2)] = color.b;
	pixel[((size_y - pos_y) * size_x * 4 + pos_x * 4+3)] = color.a;

}




int main()
{
	sf::ContextSettings setting(0,0,0);
	sf::RenderWindow window(sf::VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT)," AI ", sf::Style::Close, setting);
	sf::Event event;
	int brush_size = 5;
	bool mouse_press = false;

	//Textures for the screen and sprites
	sf::Uint8* pixels = new sf::Uint8[784 * 255 * 4];

	sf::Texture graphTexture;
	sf::Sprite graphSprite;
	//Number texture 
	sf::Uint8* numberPixels = new sf::Uint8[28 * 28 * 4];
	sf::Texture numberTex;
	sf::Sprite numberSprite;
	//Canvas for the writing text
	sf::Uint8* canvasPixel = new sf::Uint8[28 * 28 * 4];
	sf::Texture canvasTex;
	sf::Sprite canvasSprite;


	//Texts and Fonts	
	sf::Font bananaFont;
	sf::Text text("AI model",bananaFont,20);
	sf::Text textTrainNumber(" 0 ", bananaFont, 20);


	//file handling 
	fstream trainFile;
	char num[5] = {0};


	int lineNumber = 10;
	
	//Testing purpose
	//my::Point point(10,10, 5);

	sf::Vector2i mousePos;

	sf::Text textCursorPos_x("0", bananaFont, 20);
	sf::Text textCursorPos_y("0", bananaFont, 20);


	trainFile.open("res/train.csv", ios::in);



#pragma region Texture
	//Texture 
	graphTexture.create(784,255);
	graphSprite.setTexture(graphTexture);
	//graphSprite.scale(1.0f, 2.5f);
	graphSprite.setPosition(50, 200);


	//Texture of number
	numberTex.create(28, 28);
	numberTex.setSmooth(true);
	numberSprite.setTexture(numberTex);
	numberSprite.setPosition(1000, 40);
	numberSprite.setScale(10, 10);

	//Texture for canvas writing
	canvasTex.create(28, 28);
	canvasTex.setSmooth(true);
	canvasSprite.setTexture(canvasTex);
	canvasSprite.setPosition(1000, 580);
	canvasSprite.setScale(10, 10);
#pragma endregion Texture

#pragma region Text
	// Text function
	text.setPosition(SCREEN_WIDTH / 2, 0);
	textTrainNumber.setPosition(SCREEN_WIDTH / 4, 0);

	textCursorPos_x.setPosition(SCREEN_WIDTH / 2-100, 800);
	textCursorPos_y.setPosition(SCREEN_WIDTH / 2 + 100, 800);

	text.setString(" AI Modeling ");
	if (!bananaFont.loadFromFile("res/Fonts/Banana.ttf"))
	{
		std::cout << "failed to load font" << std::endl;
	}

#pragma endregion Text

#pragma region Init

	//Init stuffs 
	window.setFramerateLimit(60);

	//Main data set show 
	for (int j =0; j < 255; j++)
	{
		for (int i = 0; i < 784; i++)
		{
			PaintPixel(pixels, i, j, 784, 255,sf::Color::Black);
		}
	}

	//memset(graphPixels, 0, 784*255*4*sizeof(sf::Uint8));

	//Our canvas 
	for (int i = 0; i < 28 * 28 * 4; i += 4) {

		canvasPixel[i] = 50; // obviously, assign the values you need here to form your colorpixels[i+1] = g;
		canvasPixel[i + 1] = 50; // obviously, assign the values you need here to form your colorpixels[i+1] = g;
		canvasPixel[i + 2] = 50;
		canvasPixel[i + 3] = 50;

	}
	canvasTex.update(canvasPixel);

#pragma endregion Init

	//Main Loop for the frame by frame updated code
	while (window.isOpen())
	{
		
		
		while (window.pollEvent(event))
		{


			//Mouse Button left event	 &&   paints the canvas
			if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left))
			{
				mousePos = sf::Mouse::getPosition(window);
				//Writing On the Canvas
				if ((mousePos.x > 1000 + brush_size + 1) && (mousePos.y > 580 + brush_size + 1))
				{
					int pos_x = mousePos.x;
					int pos_y = mousePos.y;

					int pix = 0;

					pix = (((pos_y - 580) / 10) * 28 * 4) + (((pos_x - 1000) / 10) * 4);

					cout << pix << endl;

					if (pix >= 3134)
					{
						pix = 3134;
					}

					for (int j = 0; j < 2; j++)
					{

						for (int i = 0; i < 5; i++)
						{
							canvasPixel[pix + 28 * j * 4 + i + 0] = 250;
							canvasPixel[pix + 28 * j * 4 + i + 1] = 250;
							canvasPixel[pix + 28 * j * 4 + i + 2] = 250;
							canvasPixel[pix + 28 * j * 4 + i + 3] = 250;
							//for the values of canvas pixel in graph

							PaintPixel(pixels, pix/4, canvasPixel[pix], 784, 255, sf::Color::Red);
						}

					}
					canvasTex.update(canvasPixel);
					graphTexture.update(pixels);
					//update mouse cursor position
					mousePos = sf::Mouse::getPosition(window);

					textCursorPos_x.setString(to_string(mousePos.x));
					textCursorPos_y.setString(to_string(mousePos.y));

				}

			}


			//other key events  &&  extracts text from the train.csv file and displays it
			switch (event.type)
			{
				
				case sf::Event::KeyPressed:
					if (event.key.code == sf::Keyboard::A)
					{
						if (trainFile.is_open())
						{
							cout << "file is open" << endl;

							GotoLineFromBeg(trainFile, lineNumber += 1);
							//GotoLineFromCurrent(trainFile, 1);

							trainFile.getline(num, 5, ',');

							textTrainNumber.setString(num);

							for (int i = 0; i < 784 * 4; i = i + 4)
							{


								trainFile.getline(num, 5, ',');

								/*pixels[i] = std::stoi(num);
								pixels[pixels[i] * 784 * 4 + i] = 255;*/
								
								//PaintPixel(pixels, i / 4,std::stoi(num), 784, 255, sf::Color::Green);
								numberPixels[i] = std::stoi(num);
								numberPixels[i + 1] = std::stoi(num) ;
								numberPixels[i + 2] = std::stoi(num);
								numberPixels[i + 3] = std::stoi(num);
								pixels[i] = numberPixels[i];
								pixels[i+1] = numberPixels[i+1];
								pixels[i+2] = numberPixels[i+2];
								pixels[i+3] = numberPixels[i+3];

								
								//cout << std::stoi(num)<<":"<< (int)pixels[i]<<":"<<(int)numberPixels[i]<<";  ";
								
							}

						}
						
						graphTexture.update(pixels);
						numberTex.update(numberPixels);
						

					}
					if (event.key.code == sf::Keyboard::Enter)
					{
						for (int i = 0; i < 28 * 28 * 4; i += 4) {

							canvasPixel[i] = 0; // obviously, assign the values you need here to form your colorpixels[i+1] = g;
							canvasPixel[i + 1] = 0; // obviously, assign the values you need here to form your colorpixels[i+1] = g;
							canvasPixel[i + 2] = 0;
							canvasPixel[i + 3] = 0;

							memset(pixels, 0, sizeof(sf::Uint8) * 784 * 255);

						}
						canvasTex.update(canvasPixel);
						graphTexture.update(pixels);

					}
					break;



				case sf::Event::Closed:
				
					trainFile.close(); // close the file object.
					window.close();

					break;
			}

		}


	

		//Main graph 
		//PaintPixel(pixels, 10, 10, 784, 28,sf::Color::Cyan);
		//graphTexture.update(graphPixels);
		



		window.clear();

		window.draw(graphSprite);
		window.draw(text);
		
		window.draw(numberSprite);
		window.draw(textCursorPos_x);
		window.draw(textCursorPos_y);
		window.draw(canvasSprite);
		window.draw(textTrainNumber);

		window.display();		
	}


	MatrixXd A(3, 3);
	VectorXd b(3);


	A << 1, 2, 3,
		 4, 5, 6,
		 7, 8, 10;

	b << 3, 3, 4;

	VectorXd x = A.fullPivLu().solve(b);

	//cout << "The solution is:" << endl << x << endl;
	//cout << window.getSettings().antialiasingLevel << " " << window.getSettings().depthBits <<" "<< window.getSettings().stencilBits;
	return 0;
}
