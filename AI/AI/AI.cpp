#include "AI.h"
#include "My.h"

using namespace Eigen;	
using namespace std;


void GotoLine(std::fstream& file, unsigned int num) {

	file.seekg(std::ios::beg);
	for (int i = 0; i < num - 1; ++i) {
		file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}

}




int main()
{
	sf::ContextSettings setting(0,0,0);
	sf::RenderWindow window(sf::VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT)," AI ", sf::Style::Close, setting);
	sf::Event event;
	int brush_size = 5;
	bool mouse_press = false;

	//Textures for the screen and sprites
	sf::Uint8* pixels = new sf::Uint8[SCREEN_WIDTH * SCREEN_HEIGHT * 4];
	sf::Texture tex;
	sf::Sprite sprite;
	//Number texture 
	sf::Uint8* numberPixels = new sf::Uint8[28*28*4];
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

	int lineNumber = 11;
	
	//Testing purpose
	my::Point point(10,10, 5);

	sf::Vector2i mousePos;

	sf::Text textCursorPos_x("0", bananaFont, 20);
	sf::Text textCursorPos_y("0", bananaFont, 20);


	trainFile.open("res/train.csv", ios::in);



#pragma region Texture
	//Texture 
	tex.create(900, 600);
	sprite.setTexture(tex);
	sprite.setPosition(50, 50);


	//Texture of number
	numberTex.create(28, 28);
	numberTex.setSmooth(true);
	numberSprite.setTexture(numberTex);
	numberSprite.setPosition(1000, 40);
	numberSprite.setScale(10, 10);

	//Texture for canvas
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

					if (pix >= 3135)
					{
						pix = 3135;
					}

					for (int j = 0; j < 2; j++)
					{

						for (int i = 0; i < 5; i++)
						{
							canvasPixel[pix + 28 * j * 4 + i + 0] = 255;
							canvasPixel[pix + 28 * j * 4 + i + 1] = 255;
							canvasPixel[pix + 28 * j * 4 + i + 2] = 255;
							canvasPixel[pix + 28 * j * 4 + i + 3] = 255;
						}

					}
					canvasTex.update(canvasPixel);

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

							GotoLine(trainFile, lineNumber = lineNumber + 1);
							trainFile.getline(num, 5, ',');

							textTrainNumber.setString(num);

							for (int i = 0; i < 784 * 4; i = i + 4)
							{


								trainFile.getline(num, 5, ',');

								numberPixels[i] = std::stoi(num);
								numberPixels[i + 1] = std::stoi(num);
								numberPixels[i + 2] = std::stoi(num);
								numberPixels[i + 3] = std::stoi(num);


							}

						}
						numberTex.update(numberPixels);

					}
					if (event.key.code == sf::Keyboard::Enter)
					{
						for (int i = 0; i < 28 * 28 * 4; i += 4) {

							canvasPixel[i] = 50; // obviously, assign the values you need here to form your colorpixels[i+1] = g;
							canvasPixel[i + 1] = 50; // obviously, assign the values you need here to form your colorpixels[i+1] = g;
							canvasPixel[i + 2] = 50;
							canvasPixel[i + 3] = 50;

						}
						canvasTex.update(canvasPixel);
					}
					break;



				case sf::Event::Closed:
				
					trainFile.close(); // close the file object.
					window.close();

					break;
			}

		}


	

		//Main graph 
		for (int j = 0; j < 28 ; i+=1)
		{
			for (int i = j*28*4; i < (j + 1) * 28 * 4; i += 4) 
			{
				pixels[i*j + ] = 1; // obviously, assign the values you need here to form your colorpixels[i+1] = g;
				pixels[i + 1] = 100; // obviously, assign the values you need here to form your colorpixels[i+1] = g;
				pixels[i + 2] = 200;
				pixels[i + 3] = 255;

			}
		}
		tex.update(pixels);



		window.clear();

		window.draw(sprite);
		window.draw(text);
		window.draw(point.getSFML());
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
