 --data cleaning and transformation
 
-- standardised data format

-- looking at overall basic statistics of cost
select distinct max(cost) as max_cost, min(cost) as min_cost, AVG(cost) as mean_cost from portfolio..mcc;

 select (
 (select distinct max(cost) from (select top 50 percent cost from portfolio..mcc order by cost asc) as p1)
 + (select distinct min(cost) from (select top 50 percent cost from portfolio..mcc order by cost  desc) as p2)
  )/2 as median;  
 
 --cost and qualitative factors
--looking at the average store sales/cost when store has/ doesnt have a coffee bar
 select AVG(cost) as mean_cost, AVG(store_sales_in_millions) as mean_sales, coffee_bar from portfolio..mcc group by coffee_bar;
 
-- looking at the average store sales/cost when store has/ doesnt have a video store
  select AVG(cost) as mean_cost, AVG(store_sales_in_millions) as mean_sales, video_store  from portfolio..mcc group by video_store;
 
-- lookng at the average store sales/cost when store has/ doesnt have a salad bar
  select AVG(cost) as mean_cost, AVG(store_sales_in_millions) as mean_sales, salad_bar  from portfolio..mcc group by salad_bar;
 
-- looking at the average store sales/cost when store has/ doesnt have prepared food
  select AVG(cost) as mean_cost, AVG(store_sales_in_millions) as mean_sales, prepared_food  from portfolio..mcc group by prepared_food;
 

-- looking at the average store sales/cost when store has/ doesnt have a florist
  select AVG(cost) as mean_cost, AVG(store_sales_in_millions) as mean_sales, florist from portfolio..mcc group by florist;

-- looking at the average store sales/cost when store has/ doesnt have a recyclable package
  select AVG(cost) as mean_cost, AVG(store_sales_in_millions) as mean_sales, recyclable_package from portfolio..mcc group by recyclable_package;

  -- looking at the average store sales/cost when store has/ doesnt have low_fat food
  select AVG(cost) as mean_cost, AVG(store_sales_in_millions) as mean_sales, low_fat from portfolio..mcc group by low_fat;

 -- looking at the average number of cars for customers in the store group by cost and order by cost
 select AVG(avg_cars_at_home_approx_1) as ave_car_num, cost from portfolio..mcc group by cost order by cost desc;

 --looking at the average number of children at home for customers in the store group by cost and order by cost
  select AVG(num_children_at_home) as ave_children_home, cost from portfolio..mcc group by cost order by cost desc;

  --looking at the average number of children total for customers in the store group by cost and order by cost
  select AVG(total_children) as ave_children_total, cost from portfolio..mcc group by cost order by cost desc;

  --looking at the average gross weight group by cost and order by cost
  select AVG(gross_weight) as ave_gross_weight, cost from portfolio..mcc group by cost order by cost desc;

   --looking at the average units per case group by cost and order by cost
  select AVG(units_per_case) as ave_units_per_case, cost from portfolio..mcc group by cost order by cost desc;

   --looking at the average store area group by cost and order by cost
  select AVG(store_sqft) as ave_store_area, cost from portfolio..mcc group by cost order by cost desc;
