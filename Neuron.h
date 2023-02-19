#include <SFML/Main.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

namespace my
{
	class Point
	{
	private:
		sf::CircleShape point;
		

	public:
		Point();
		Point(float pos_x, float pos_y, float radius = 3.0f);

		sf::CircleShape& getSFML()
		{
			return point;
		}

		void setColor(sf::Color color);
	};
	



}