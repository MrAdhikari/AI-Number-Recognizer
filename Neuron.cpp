#include "My.h"
#include <math.h>

my::Point::Point()
{
	
	point.setFillColor(sf::Color::White);
	point.setRadius(3.0f);
}

void my::Point::setColor(sf::Color color)
{
	point.setFillColor(color);
}

my::Point::Point(float pos_x, float pos_y, float radius /*= 3.0f*/)
{
	point.setFillColor(sf::Color::White);
	point.setPosition(pos_x, pos_y);
	point.setRadius(radius);
}

